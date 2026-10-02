using System.Globalization;
using MediatR;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Models;

public sealed record StratumCoinDto(Guid CoinId, string Code);

public sealed record StratumPoolDto(
    Guid PoolId,
    string SystemPoolId,
    string Name,
    string StratumEndpoint,
    string? PayoutAddress,
    IReadOnlyList<StratumCoinDto> Coins);

public sealed record StratumConfigDto(string Version, IReadOnlyList<StratumPoolDto> Pools);

public sealed record StratumWorkerDto(
    string WorkerId,
    Guid UserId,
    Guid UserHardwareId,
    Guid UserHardwareMinerId,
    Guid PoolId,
    string StratumEndpoint,
    string? PayoutAddress,
    Guid CoinId,
    string CoinCode);

public sealed record GetStratumConfigQuery : IRequest<StratumConfigDto>;

/// <summary>
/// Builds the proxy's routing table: active pools that have a stratum endpoint and at least one
/// active coin. Pools missing either are omitted rather than published half-configured.
/// </summary>
internal sealed class GetStratumConfigQueryHandler(IPoolRepository pools, ICoinRepository coins)
    : IRequestHandler<GetStratumConfigQuery, StratumConfigDto>
{
    public async Task<StratumConfigDto> Handle(
        GetStratumConfigQuery request,
        CancellationToken cancellationToken)
    {
        var allPools = await pools.ListAsync(cancellationToken);
        var allDetails = await pools.ListDetailsAsync(cancellationToken);
        var poolCoins = await pools.ListPoolCoinsAsync(cancellationToken);
        var allCoins = await coins.ListAsync(cancellationToken);

        var activeCoins = allCoins
            .Where(coin => coin.Status == MiningStatus.Active)
            .ToDictionary(coin => coin.Id, coin => coin.Code);

        var detailsByPool = allDetails.ToDictionary(details => details.PoolId);
        var coinsByPool = poolCoins
            .GroupBy(poolCoin => poolCoin.PoolId)
            .ToDictionary(group => group.Key, group => group.Select(poolCoin => poolCoin.CoinId).ToList());

        var published = new List<StratumPoolDto>();
        var latestChange = DateTimeOffset.UnixEpoch;

        foreach (var pool in allPools.Where(pool => pool.Status == MiningStatus.Active))
        {
            if (!detailsByPool.TryGetValue(pool.Id, out var details)
                || string.IsNullOrWhiteSpace(details.StratumEndpoint))
            {
                continue;
            }

            var supportedCoins = (coinsByPool.GetValueOrDefault(pool.Id) ?? [])
                .Where(activeCoins.ContainsKey)
                .Select(coinId => new StratumCoinDto(coinId, activeCoins[coinId]))
                .ToList();

            if (supportedCoins.Count == 0)
            {
                continue;
            }

            published.Add(new StratumPoolDto(
                pool.Id,
                pool.SystemPoolId,
                pool.Name,
                details.StratumEndpoint,
                details.PayoutAddress,
                supportedCoins));

            latestChange = pool.UpdatedAt > latestChange ? pool.UpdatedAt : latestChange;
            latestChange = details.UpdatedAt > latestChange ? details.UpdatedAt : latestChange;
        }

        var version = latestChange == DateTimeOffset.UnixEpoch
            ? "0"
            : latestChange.ToUnixTimeMilliseconds().ToString(CultureInfo.InvariantCulture);

        return new StratumConfigDto(version, published);
    }
}

public sealed record ResolveStratumWorkerQuery(string WorkerId) : IRequest<StratumWorkerDto?>;

/// <summary>
/// Resolves a worker identity to the session it is currently running. Returns null when the
/// worker has no active session, which the proxy treats as "not allowed to mine".
/// </summary>
internal sealed class ResolveStratumWorkerQueryHandler(
    IMiningSessionRepository sessions,
    IUserHardwareRepository hardware,
    IPoolRepository pools,
    ICoinRepository coins) : IRequestHandler<ResolveStratumWorkerQuery, StratumWorkerDto?>
{
    public async Task<StratumWorkerDto?> Handle(
        ResolveStratumWorkerQuery request,
        CancellationToken cancellationToken)
    {
        var session = await sessions.GetActiveByWorkerIdentifierAsync(request.WorkerId, cancellationToken);
        if (session is null)
        {
            return null;
        }

        var device = await hardware.GetByIdAsync(session.UserHardwareId, cancellationToken);
        if (device is null)
        {
            return null;
        }

        var details = await pools.GetDetailsAsync(session.PoolId, cancellationToken);
        if (details is null || string.IsNullOrWhiteSpace(details.StratumEndpoint))
        {
            return null;
        }

        var coin = await coins.GetByIdAsync(session.CoinId, cancellationToken);
        if (coin is null)
        {
            return null;
        }

        return new StratumWorkerDto(
            session.WorkerIdentifier,
            device.UserId,
            device.Id,
            session.Id,
            session.PoolId,
            details.StratumEndpoint,
            details.PayoutAddress,
            coin.Id,
            coin.Code);
    }
}
