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
    string Status,
    DateTimeOffset StartedAt);

/// <summary>
/// <paramref name="Reason"/> is one of <c>connection-failure</c>, <c>driver-crash</c>,
/// <c>miner-closed</c>, <c>user-triggered</c> or <c>unknown</c>.
/// </summary>
public sealed record StopMiningRequest(Guid HardwareId, string Reason);

// --- WebSocket protocol (/ws/mining) -----------------------------------------------------

/// <summary>Client heartbeat sent every few seconds while a device is mining.</summary>
public sealed record MiningHeartbeatMessage(string Type, Guid HardwareId);

public sealed record MiningHeartbeatAck(
    string Type,
    bool SessionFound,
    string? Status,
    bool Resumed,
    DateTimeOffset At);
