using FluentValidation;
using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;
using TokenMiner.Domain.Users;

namespace TokenMiner.Application.Mining.Shares;

public sealed record RecordShareCommand(
    string WorkerIdentifier,
    Guid PoolId,
    Guid CoinId,
    string ShareIdentifier,
    string? JobId,
    string? Nonce,
    string? Extranonce,
    decimal? Difficulty,
    string? Target,
    string? ResultHash,
    DateTimeOffset ShareTimestamp,
    decimal? CoinValue,
    string? PoolResponse) : IRequest<RecordedShareDto>;

public sealed class RecordShareCommandValidator : AbstractValidator<RecordShareCommand>
{
    public RecordShareCommandValidator()
    {
        RuleFor(x => x.WorkerIdentifier).NotEmpty().MaximumLength(64);
        RuleFor(x => x.PoolId).NotEmpty();
        RuleFor(x => x.CoinId).NotEmpty();
        RuleFor(x => x.ShareIdentifier).NotEmpty().MaximumLength(256);
        RuleFor(x => x.JobId).MaximumLength(128);
        RuleFor(x => x.Nonce).MaximumLength(128);
        RuleFor(x => x.Extranonce).MaximumLength(128);
        RuleFor(x => x.Target).MaximumLength(128);
        RuleFor(x => x.ResultHash).MaximumLength(256);
        RuleFor(x => x.CoinValue).GreaterThanOrEqualTo(0).When(x => x.CoinValue is not null);
    }
}

/// <summary>
/// Records a pool-accepted share. Idempotent on (pool, share identifier): replaying the same
/// share returns the stored row instead of inserting a second one.
///
/// The proxy never resolves the worker, so the request carries only the pool-facing worker id —
/// the device's hardware id in "N" form. A device the API has not seen is created (with a
/// placeholder owner, since the hardware row needs a user) so the share is never lost.
/// </summary>
internal sealed class RecordShareCommandHandler(
    IShareRepository shares,
    IUserRepository users,
    IUserHardwareRepository hardware,
    IMiningSessionRepository sessions,
    IMiningAlgoRepository algorithms,
    IPoolRepository pools,
    ICoinRepository coins,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<RecordShareCommand, RecordedShareDto>
{
    public async Task<RecordedShareDto> Handle(RecordShareCommand request, CancellationToken cancellationToken)
    {
        var existing = await shares.GetByPoolAndIdentifierAsync(
            request.PoolId,
            request.ShareIdentifier,
            cancellationToken);

        if (existing is not null)
        {
            return existing.ToDto(alreadyRecorded: true);
        }

        if (!Guid.TryParse(request.WorkerIdentifier, out var hardwareId))
        {
            throw new BadRequestException("The worker id is not a hardware id.");
        }

        if (!await pools.SupportsCoinAsync(request.PoolId, request.CoinId, cancellationToken))
        {
            throw new BadRequestException("The pool does not support the requested coin.");
        }

        var coin = await coins.GetByIdAsync(request.CoinId, cancellationToken)
            ?? throw new BadRequestException("Coin not found.");

        var now = timeProvider.GetUtcNow();

        var device = await ResolveDeviceAsync(hardwareId, now, cancellationToken);
        var session = await ResolveSessionAsync(request, device, now, cancellationToken);

        var share = new UserMiningShare(
            Guid.NewGuid(),
            device.UserId,
            device.Id,
            session.Id,
            request.PoolId,
            request.CoinId,
            request.ShareIdentifier,
            session.WorkerIdentifier,
            request.JobId,
            request.Nonce,
            request.Extranonce,
            request.Difficulty,
            request.Target,
            request.ResultHash,
            request.ShareTimestamp,
            request.CoinValue,
            coin.LastKnownUsdValue,
            request.PoolResponse,
            now);

        shares.Add(share);

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return share.ToDto();
    }

    /// <summary>
    /// The device the worker names. When the API has never seen it — the miner was proxied without
    /// going through <c>POST /api/mining/start</c> — a placeholder owner and the device are created
    /// so the share is attributed rather than dropped.
    /// </summary>
    private async Task<UserHardware> ResolveDeviceAsync(
        Guid hardwareId,
        DateTimeOffset now,
        CancellationToken cancellationToken)
    {
        var device = await hardware.GetByHardwareIdAsync(hardwareId, cancellationToken);

        if (device is not null)
        {
            device.Touch(now);
            return device;
        }

        var email = $"worker-{Guid.NewGuid():N}@placeholder.tokenminer.local";
        var owner = new User(
            Guid.NewGuid(),
            email,
            email.ToUpperInvariant(),
            passwordHash: null,
            displayName: "Unclaimed worker",
            now);

        users.Add(owner);

        var created = new UserHardware(Guid.NewGuid(), owner.Id, hardwareId, name: null, now);
        hardware.Add(created);

        return created;
    }

    /// <summary>
    /// The device's live session. There may be none when the miner is proxied without going through
    /// <c>POST /api/mining/start</c>, so one is opened with an active algorithm.
    /// </summary>
    private async Task<UserHardwareMiner> ResolveSessionAsync(
        RecordShareCommand request,
        UserHardware device,
        DateTimeOffset now,
        CancellationToken cancellationToken)
    {
        var session = await sessions.GetActiveByHardwareAsync(device.Id, cancellationToken);

        if (session is not null)
        {
            session.Touch(now);
            return session;
        }

        var algorithm = (await algorithms.ListAsync(cancellationToken))
            .Where(candidate => candidate.Status == MiningStatus.Active)
            .OrderByDescending(candidate => candidate.Priority)
            .ThenBy(candidate => candidate.Code, StringComparer.Ordinal)
            .FirstOrDefault()
            ?? throw new ConflictException("No active mining algorithm is configured.");

        var created = new UserHardwareMiner(
            Guid.NewGuid(),
            device.Id,
            request.PoolId,
            request.CoinId,
            algorithm.Id,
            request.WorkerIdentifier,
            now);

        sessions.Add(created);

        return created;
    }
}
