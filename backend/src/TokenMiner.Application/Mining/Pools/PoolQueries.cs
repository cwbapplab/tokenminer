using MediatR;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;

namespace TokenMiner.Application.Mining.Pools;

public sealed record GetPoolQuery(Guid Id) : IRequest<PoolDto>;

internal sealed class GetPoolQueryHandler(IPoolRepository pools) : IRequestHandler<GetPoolQuery, PoolDto>
{
    public async Task<PoolDto> Handle(GetPoolQuery request, CancellationToken cancellationToken)
    {
        var pool = await pools.GetByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("Pool not found.");

        var details = await pools.GetDetailsAsync(pool.Id, cancellationToken);
        var poolCoins = await pools.ListPoolCoinsAsync(pool.Id, cancellationToken);

        return pool.ToDto(details, poolCoins.Select(poolCoin => poolCoin.CoinId).ToList());
    }
}

public sealed record ListPoolsQuery : IRequest<IReadOnlyList<PoolDto>>;

internal sealed class ListPoolsQueryHandler(IPoolRepository pools)
    : IRequestHandler<ListPoolsQuery, IReadOnlyList<PoolDto>>
{
    public async Task<IReadOnlyList<PoolDto>> Handle(
        ListPoolsQuery request,
        CancellationToken cancellationToken)
    {
        // Three queries total rather than one per pool.
        var all = await pools.ListAsync(cancellationToken);
        var details = await pools.ListDetailsAsync(cancellationToken);
        var poolCoins = await pools.ListPoolCoinsAsync(cancellationToken);

        var detailsByPool = details.ToDictionary(item => item.PoolId);
        var coinsByPool = poolCoins
            .GroupBy(poolCoin => poolCoin.PoolId)
            .ToDictionary(
                group => group.Key,
                group => (IReadOnlyList<Guid>)group.Select(poolCoin => poolCoin.CoinId).ToList());

        return all
            .Select(pool => pool.ToDto(
                detailsByPool.GetValueOrDefault(pool.Id),
                coinsByPool.GetValueOrDefault(pool.Id) ?? []))
            .ToList();
    }
}
