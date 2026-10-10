using System.Text.Json;

namespace TokenMiner.Contracts.Mining;

/// <summary>
/// Starts mining on a device. <paramref name="PoolId"/> is an optional override; when omitted
/// the API selects an active pool deterministically.
/// </summary>
public sealed record StartMiningRequest(Guid HardwareId, Guid? PoolId);

public sealed record MiningSessionResponse(
    Guid SessionId,
    Guid PoolId,
    string PoolName,
    Guid CoinId,
    string CoinCode,
    Guid AlgorithmId,
    string AlgorithmCode,
    string WorkerId,
    string MinerCommand,
    /// <summary>
    /// The algorithm configuration with the session's placeholders rendered — the structured
    /// form clients read directly (algo, endpoint, wallet, workerId, coin).
    /// </summary>
    JsonElement MinerConfig,
    /// <summary>
    /// The public Stratum endpoint (host:port) the client must connect to — always the
    /// TokenMiner proxy, never the pool. Clients must not connect to a pool directly.
    /// </summary>
    string StratumEndpoint,
    string Status,
    DateTimeOffset StartedAt);

/// <summary>
/// <paramref name="Reason"/> is one of <c>connection-failure</c>, <c>driver-crash</c>,
/// <c>miner-closed</c>, <c>user-triggered</c> or <c>unknown</c>.
/// </summary>
public sealed record StopMiningRequest(Guid HardwareId, string Reason);
