using System.Text.Json;
using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Application.Mining.Services;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Sessions;

public sealed record StartMiningCommand(Guid UserId, Guid HardwareId, Guid? PoolId) : IRequest<MiningSessionDto>;

public sealed class StartMiningCommandValidator : AbstractValidator<StartMiningCommand>
{
    public StartMiningCommandValidator()
    {
        RuleFor(x => x.UserId).NotEmpty();
        RuleFor(x => x.HardwareId).NotEmpty();
    }
}

/// <summary>
/// Starts a mining session for one of the caller's devices: resolves (or registers) the
/// hardware, picks a pool/coin/algorithm, records the session and returns the launch command.
/// </summary>
internal sealed class StartMiningCommandHandler(
    IUserHardwareRepository hardware,
    IUserRepository users,
    IMiningSessionRepository sessions,
    IPoolRepository pools,
    ICoinRepository coins,
    IMiningAlgoRepository algorithms,
    IMiningLogRepository logs,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider,
    MiningOptions miningOptions) : IRequestHandler<StartMiningCommand, MiningSessionDto>
{
    public async Task<MiningSessionDto> Handle(StartMiningCommand request, CancellationToken cancellationToken)
    {
        var now = timeProvider.GetUtcNow();

        var device = await hardware.GetByHardwareIdAsync(request.HardwareId, cancellationToken);

        if (device is null)
        {
            device = new UserHardware(Guid.NewGuid(), request.UserId, request.HardwareId, name: null, now);
            hardware.Add(device);
        }
        else if (device.UserId != request.UserId)
        {
            // The hardware id is the device's identity. One first seen through a share carries a
            // placeholder owner, so the real account starting mining on it takes ownership; a
            // device already owned by another account is refused.
            var owner = await users.GetByIdAsync(device.UserId, cancellationToken);

            if (owner is { PasswordHash: null, GoogleSub: null })
            {
                device.AssignOwner(request.UserId, now);
            }
            else
            {
                throw new ConflictException("This device is registered to another account.");
            }
        }
        else
        {
            device.Touch(now);
        }

        if (await sessions.GetActiveByHardwareAsync(device.Id, cancellationToken) is not null)
        {
            throw new ConflictException("Mining is already active for this device.");
        }

        var (pool, poolDetails, coin) = await SelectPoolAsync(request.PoolId, cancellationToken);
        var (algorithm, configuration) = await SelectAlgorithmAsync(coin, cancellationToken);

        // The worker identity is always derived server-side; a client can never choose it. The
        // hardware id in "N" form is 32 chars, which is the pool's cap for a worker name.
        var workerIdentifier = device.HardwareId.ToString("N");

        var session = new UserHardwareMiner(
            Guid.NewGuid(),
            device.Id,
            pool.Id,
            coin.Id,
            algorithm.Id,
            workerIdentifier,
            now);

        sessions.Add(session);

        var upstreamStratumHost = string.IsNullOrWhiteSpace(poolDetails.StratumEndpoint)
            ? StripScheme(poolDetails.BaseUrl)
            : poolDetails.StratumEndpoint;

        // Without a configured proxy endpoint the command points straight at the pool.
        var publicStratumHost = string.IsNullOrWhiteSpace(miningOptions.PublicStratumEndpoint)
            ? upstreamStratumHost
            : miningOptions.PublicStratumEndpoint;

        var minerCommand = MiningCommandRenderer.Render(configuration.Command, new Dictionary<string, string>
        {
            ["coin"] = coin.Code,
            ["coinName"] = coin.Name,
            ["algo"] = algorithm.Code,
            ["poolUrl"] = poolDetails.BaseUrl,
            ["poolHost"] = StripScheme(poolDetails.BaseUrl),
            ["stratumHost"] = publicStratumHost,
            ["upstreamStratumHost"] = upstreamStratumHost,
            ["wallet"] = poolDetails.PayoutAddress ?? string.Empty,
            ["workerId"] = workerIdentifier,
            ["hardwareId"] = device.HardwareId.ToString(),
        });

        logs.Add(new MiningLog(
            Guid.NewGuid(),
            request.UserId,
            device.Id,
            session.Id,
            pool.Id,
            coin.Id,
            MiningEventType.MiningStarted,
            reason: null,
            metadata: JsonSerializer.Serialize(new
            {
                poolId = pool.Id,
                coinId = coin.Id,
                algorithmId = algorithm.Id,
                workerId = workerIdentifier,
            }),
            now));

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return new MiningSessionDto(
            session.Id,
            pool.Id,
            pool.Name,
            coin.Id,
            coin.Code,
            algorithm.Id,
            algorithm.Code,
            workerIdentifier,
            minerCommand,
            session.Status.ToDbValue(),
            session.StartedAt);
    }

    private async Task<(Pool Pool, PoolDetails Details, Coin Coin)> SelectPoolAsync(
        Guid? poolId,
        CancellationToken cancellationToken)
    {
        var activePools = (await pools.ListAsync(cancellationToken))
            .Where(pool => pool.Status == MiningStatus.Active)
            .OrderBy(pool => pool.Name, StringComparer.Ordinal)
            .ToList();

        if (poolId is not null)
        {
            var requested = activePools.FirstOrDefault(pool => pool.Id == poolId.Value)
                ?? throw new BadRequestException($"Pool '{poolId}' is unknown or not active.");

            var requestedDetails = await pools.GetDetailsAsync(requested.Id, cancellationToken)
                ?? throw new BadRequestException($"Pool '{poolId}' has no connection configuration.");

            var requestedCoin = await coins.GetByIdAsync(requestedDetails.CoinId, cancellationToken);
            if (requestedCoin is null || requestedCoin.Status != MiningStatus.Active)
            {
                throw new BadRequestException($"Pool '{poolId}' references a coin that is not active.");
            }

            return (requested, requestedDetails, requestedCoin);
        }

        // Deterministic default: the first active pool (by name) with an active coin.
        foreach (var candidate in activePools)
        {
            var details = await pools.GetDetailsAsync(candidate.Id, cancellationToken);
            if (details is null)
            {
                continue;
            }

            var coin = await coins.GetByIdAsync(details.CoinId, cancellationToken);
            if (coin is null || coin.Status != MiningStatus.Active)
            {
                continue;
            }

            return (candidate, details, coin);
        }

        throw new ConflictException("No active pool with an active coin is configured.");
    }

    private async Task<(MiningAlgo Algorithm, MiningAlgoConfiguration Configuration)> SelectAlgorithmAsync(
        Coin coin,
        CancellationToken cancellationToken)
    {
        var candidates = (await algorithms.ListAsync(cancellationToken))
            .Where(algorithm => algorithm.Status == MiningStatus.Active)
            .OrderByDescending(algorithm => algorithm.Priority)
            .ThenBy(algorithm => algorithm.Code, StringComparer.Ordinal);

        foreach (var algorithm in candidates)
        {
            if (!MiningAlgoConfiguration.TryParse(algorithm.Configuration, out var configuration))
            {
                continue;
            }

            if (!configuration.SupportsCoin(coin.Code))
            {
                continue;
            }

            return (algorithm, configuration);
        }

        throw new ConflictException($"No active mining algorithm is configured for coin '{coin.Code}'.");
    }

    private static string StripScheme(string url)
    {
        var separator = url.IndexOf("://", StringComparison.Ordinal);
        return separator >= 0 ? url[(separator + 3)..] : url;
    }
}
