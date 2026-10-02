using FluentValidation;
using MediatR;
using TokenMiner.Application.Common.Abstractions;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Pools;

/// <summary>Connection and payout configuration supplied alongside a pool.</summary>
public sealed record PoolDetailsInput(
    string BaseUrl,
    string? StatusEndpoint,
    string? StratumEndpoint,
    Guid CoinId,
    string PayoutMode,
    string? PayoutAddress,
    string? PayoutNetwork);

public sealed class PoolDetailsInputValidator : AbstractValidator<PoolDetailsInput>
{
    public PoolDetailsInputValidator()
    {
        RuleFor(x => x.BaseUrl).NotEmpty().MaximumLength(512);
        RuleFor(x => x.StatusEndpoint).MaximumLength(512);
        RuleFor(x => x.StratumEndpoint).MaximumLength(256);
        RuleFor(x => x.CoinId).NotEmpty();
        RuleFor(x => x.PayoutMode)
            .Must(value => PayoutModeDbValues.TryParseDbValue(value, out _))
            .WithMessage("PayoutMode must be 'managed_request', 'native_auto_payout' or 'direct_to_wallet'.");
        RuleFor(x => x.PayoutAddress).MaximumLength(256);
        RuleFor(x => x.PayoutNetwork).MaximumLength(64);

        // Withdrawing a pool balance ourselves means the pool has no destination wallet of its own.
        RuleFor(x => x.PayoutAddress)
            .NotEmpty()
            .When(x => !string.Equals(x.PayoutMode, PayoutModeDbValues.ManagedRequest, StringComparison.OrdinalIgnoreCase))
            .WithMessage("PayoutAddress is required unless the payout mode is 'managed_request'.");
    }
}

internal static class PoolCoinGuard
{
    public static async Task EnsureExistAsync(
        ICoinRepository coins,
        IEnumerable<Guid> coinIds,
        CancellationToken cancellationToken)
    {
        foreach (var coinId in coinIds.Distinct())
        {
            if (await coins.GetByIdAsync(coinId, cancellationToken) is null)
            {
                throw new BadRequestException($"Coin '{coinId}' does not exist.");
            }
        }
    }
}

public sealed record CreatePoolCommand(
    string SystemPoolId,
    string Name,
    string Provider,
    PoolDetailsInput Details,
    IReadOnlyList<Guid> CoinIds) : IRequest<PoolDto>;

public sealed class CreatePoolCommandValidator : AbstractValidator<CreatePoolCommand>
{
    public CreatePoolCommandValidator()
    {
        RuleFor(x => x.SystemPoolId).NotEmpty().MaximumLength(128);
        RuleFor(x => x.Name).NotEmpty().MaximumLength(128);
        RuleFor(x => x.Provider).NotEmpty().MaximumLength(64);
        RuleFor(x => x.Details).NotNull().SetValidator(new PoolDetailsInputValidator());
        RuleFor(x => x.CoinIds).NotNull().NotEmpty();
    }
}

internal sealed class CreatePoolCommandHandler(
    IPoolRepository pools,
    ICoinRepository coins,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<CreatePoolCommand, PoolDto>
{
    public async Task<PoolDto> Handle(CreatePoolCommand request, CancellationToken cancellationToken)
    {
        var systemPoolId = request.SystemPoolId.Trim();

        if (await pools.GetBySystemPoolIdAsync(systemPoolId, cancellationToken) is not null)
        {
            throw new ConflictException($"A pool with system id '{systemPoolId}' already exists.");
        }

        await PoolCoinGuard.EnsureExistAsync(coins, request.CoinIds.Append(request.Details.CoinId), cancellationToken);

        var now = timeProvider.GetUtcNow();

        var pool = new Pool(Guid.NewGuid(), systemPoolId, request.Name.Trim(), request.Provider.Trim().ToLowerInvariant(), now);
        var details = new PoolDetails(
            Guid.NewGuid(),
            pool.Id,
            request.Details.BaseUrl.Trim(),
            request.Details.StatusEndpoint?.Trim(),
            request.Details.StratumEndpoint?.Trim(),
            request.Details.CoinId,
            PayoutModeDbValues.FromDbValue(request.Details.PayoutMode),
            request.Details.PayoutAddress?.Trim(),
            request.Details.PayoutNetwork?.Trim(),
            now);

        pools.Add(pool);
        pools.AddDetails(details);

        foreach (var coinId in request.CoinIds.Distinct())
        {
            pools.AddPoolCoin(new PoolCoin(Guid.NewGuid(), pool.Id, coinId));
        }

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return pool.ToDto(details, request.CoinIds.Distinct().ToList());
    }
}

public sealed record UpdatePoolCommand(
    Guid Id,
    string Name,
    string Provider,
    string Status,
    PoolDetailsInput Details,
    IReadOnlyList<Guid> CoinIds) : IRequest<PoolDto>;

public sealed class UpdatePoolCommandValidator : AbstractValidator<UpdatePoolCommand>
{
    public UpdatePoolCommandValidator()
    {
        RuleFor(x => x.Id).NotEmpty();
        RuleFor(x => x.Name).NotEmpty().MaximumLength(128);
        RuleFor(x => x.Provider).NotEmpty().MaximumLength(64);
        RuleFor(x => x.Status)
            .Must(value => MiningStatusDbValues.TryParseDbValue(value, out _))
            .WithMessage("Status must be 'active' or 'disabled'.");
        RuleFor(x => x.Details).NotNull().SetValidator(new PoolDetailsInputValidator());
        RuleFor(x => x.CoinIds).NotNull().NotEmpty();
    }
}

internal sealed class UpdatePoolCommandHandler(
    IPoolRepository pools,
    ICoinRepository coins,
    IUnitOfWork unitOfWork,
    TimeProvider timeProvider) : IRequestHandler<UpdatePoolCommand, PoolDto>
{
    public async Task<PoolDto> Handle(UpdatePoolCommand request, CancellationToken cancellationToken)
    {
        var pool = await pools.GetByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("Pool not found.");

        await PoolCoinGuard.EnsureExistAsync(coins, request.CoinIds.Append(request.Details.CoinId), cancellationToken);

        var now = timeProvider.GetUtcNow();

        pool.UpdateDetails(
            request.Name.Trim(),
            request.Provider.Trim().ToLowerInvariant(),
            MiningStatusDbValues.FromDbValue(request.Status),
            now);

        var details = await pools.GetDetailsAsync(pool.Id, cancellationToken);
        if (details is null)
        {
            details = new PoolDetails(
                Guid.NewGuid(),
                pool.Id,
                request.Details.BaseUrl.Trim(),
                request.Details.StatusEndpoint?.Trim(),
                request.Details.StratumEndpoint?.Trim(),
                request.Details.CoinId,
                PayoutModeDbValues.FromDbValue(request.Details.PayoutMode),
                request.Details.PayoutAddress?.Trim(),
                request.Details.PayoutNetwork?.Trim(),
                now);

            pools.AddDetails(details);
        }
        else
        {
            details.Update(
                request.Details.BaseUrl.Trim(),
                request.Details.StatusEndpoint?.Trim(),
                request.Details.StratumEndpoint?.Trim(),
                request.Details.CoinId,
                PayoutModeDbValues.FromDbValue(request.Details.PayoutMode),
                request.Details.PayoutAddress?.Trim(),
                request.Details.PayoutNetwork?.Trim(),
                now);
        }

        var requested = request.CoinIds.Distinct().ToHashSet();
        var existing = await pools.ListPoolCoinsAsync(pool.Id, cancellationToken);
        var existingIds = existing.Select(poolCoin => poolCoin.CoinId).ToHashSet();

        var toRemove = existing.Where(poolCoin => !requested.Contains(poolCoin.CoinId)).ToList();
        if (toRemove.Count > 0)
        {
            pools.RemovePoolCoins(toRemove);
        }

        foreach (var coinId in requested.Where(coinId => !existingIds.Contains(coinId)))
        {
            pools.AddPoolCoin(new PoolCoin(Guid.NewGuid(), pool.Id, coinId));
        }

        await unitOfWork.SaveChangesAsync(cancellationToken);

        return pool.ToDto(details, requested.ToList());
    }
}
