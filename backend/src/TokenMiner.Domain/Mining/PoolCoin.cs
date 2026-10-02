namespace TokenMiner.Domain.Mining;

/// <summary>Association between a pool and a coin it supports.</summary>
public sealed class PoolCoin
{
    private PoolCoin()
    {
    }

    public PoolCoin(Guid id, Guid poolId, Guid coinId)
    {
        Id = id;
        PoolId = poolId;
        CoinId = coinId;
    }

    public Guid Id { get; private set; }

    public Guid PoolId { get; private set; }

    public Guid CoinId { get; private set; }
}
