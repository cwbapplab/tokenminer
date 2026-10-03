using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

namespace TokenMiner.Application.Mining.Abstractions;

public interface ICoinRepository
{
    Task<Coin?> GetByIdAsync(Guid id, CancellationToken cancellationToken);

    Task<Coin?> GetByCodeAsync(string code, CancellationToken cancellationToken);

    Task<IReadOnlyList<Coin>> ListAsync(CancellationToken cancellationToken);

    void Add(Coin coin);

    void AddPriceHistory(CoinPriceHistory pricePoint);
}

public interface IPoolRepository
{
    Task<Pool?> GetByIdAsync(Guid id, CancellationToken cancellationToken);

    Task<Pool?> GetBySystemPoolIdAsync(string systemPoolId, CancellationToken cancellationToken);

    Task<IReadOnlyList<Pool>> ListAsync(CancellationToken cancellationToken);

    Task<PoolDetails?> GetDetailsAsync(Guid poolId, CancellationToken cancellationToken);

    Task<IReadOnlyList<PoolDetails>> ListDetailsAsync(CancellationToken cancellationToken);

    Task<IReadOnlyList<PoolCoin>> ListPoolCoinsAsync(CancellationToken cancellationToken);

    Task<IReadOnlyList<PoolCoin>> ListPoolCoinsAsync(Guid poolId, CancellationToken cancellationToken);

    /// <summary>Whether the pool advertises support for the coin, i.e. a <c>pool_coins</c> row exists.</summary>
    Task<bool> SupportsCoinAsync(Guid poolId, Guid coinId, CancellationToken cancellationToken);

    void Add(Pool pool);

    void AddDetails(PoolDetails details);

    void AddPoolCoin(PoolCoin poolCoin);

    void RemovePoolCoins(IEnumerable<PoolCoin> poolCoins);
}

public interface IMiningAlgoRepository
{
    Task<MiningAlgo?> GetByIdAsync(Guid id, CancellationToken cancellationToken);

    Task<MiningAlgo?> GetByCodeAsync(string code, CancellationToken cancellationToken);

    Task<IReadOnlyList<MiningAlgo>> ListAsync(CancellationToken cancellationToken);

    void Add(MiningAlgo algo);
}

public interface IUserHardwareRepository
{
    Task<UserHardware?> GetByIdAsync(Guid id, CancellationToken cancellationToken);

    Task<IReadOnlyList<UserHardware>> ListByIdsAsync(
        IEnumerable<Guid> ids,
        CancellationToken cancellationToken);

    Task<UserHardware?> GetByUserAndHardwareIdAsync(
        Guid userId,
        Guid hardwareId,
        CancellationToken cancellationToken);

    /// <summary>
    /// The device a pool-facing worker id names. A client-generated hardware GUID is unique in
    /// practice, so the hardware id alone resolves the device (and through it the user).
    /// </summary>
    Task<UserHardware?> GetByHardwareIdAsync(Guid hardwareId, CancellationToken cancellationToken);

    Task<IReadOnlyList<UserHardware>> ListByUserAsync(Guid userId, CancellationToken cancellationToken);

    /// <summary>How many devices each of the given users owns, keyed by user id.</summary>
    Task<IReadOnlyDictionary<Guid, int>> CountByUsersAsync(
        IEnumerable<Guid> userIds,
        CancellationToken cancellationToken);

    void Add(UserHardware hardware);
}

public interface IMiningSessionRepository
{
    Task<UserHardwareMiner?> GetByIdAsync(Guid id, CancellationToken cancellationToken);

    /// <summary>The single in-flight (running or paused) session for a device, if any.</summary>
    Task<UserHardwareMiner?> GetActiveByHardwareAsync(Guid userHardwareId, CancellationToken cancellationToken);

    /// <summary>
    /// The in-flight session owning a worker identity. The worker id is treated as an opaque
    /// key, so the proxy never has to parse it.
    /// </summary>
    Task<UserHardwareMiner?> GetActiveByWorkerIdentifierAsync(
        string workerIdentifier,
        CancellationToken cancellationToken);

    /// <summary>The in-flight (running or paused) sessions for the given devices.</summary>
    Task<IReadOnlyList<UserHardwareMiner>> ListActiveByHardwareIdsAsync(
        IEnumerable<Guid> hardwareIds,
        CancellationToken cancellationToken);

    /// <summary>Running sessions whose last heartbeat is older than <paramref name="threshold"/>.</summary>
    Task<IReadOnlyList<UserHardwareMiner>> ListRunningIdleSinceAsync(
        DateTimeOffset threshold,
        CancellationToken cancellationToken);

    Task<int> CountByStatusAsync(MiningSessionStatus status, CancellationToken cancellationToken);

    void Add(UserHardwareMiner session);
}

public interface IMiningLogRepository
{
    void Add(MiningLog log);
}
