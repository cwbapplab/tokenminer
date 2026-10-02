namespace TokenMiner.Application.Mining;

/// <summary>Bound from the <c>Mining</c> configuration section.</summary>
public sealed class MiningOptions
{
    public const string SectionName = "Mining";

    /// <summary>How long a session may go without a heartbeat before the watchdog pauses it.</summary>
    public int HeartbeatGraceSeconds { get; set; } = 10;

    /// <summary>
    /// Stratum endpoint clients should connect to, i.e. the proxy. When empty, generated miner
    /// commands point straight at the pool, bypassing the proxy.
    /// </summary>
    public string PublicStratumEndpoint { get; set; } = string.Empty;

    /// <summary>How often the watchdog looks for stale sessions.</summary>
    public int WatchdogIntervalSeconds { get; set; } = 5;

    /// <summary>How often the reward processor advances pending shares.</summary>
    public int ShareProcessorIntervalSeconds { get; set; } = 60;

    /// <summary>How many pending shares one processor pass claims.</summary>
    public int ShareProcessorBatchSize { get; set; } = 200;

    /// <summary>How often the analytics rollup is rebuilt.</summary>
    public int StatisticsIntervalSeconds { get; set; } = 900;

    /// <summary>Shared secret the stratum proxy signs internal requests with.</summary>
    public string ServiceSharedSecret { get; set; } = string.Empty;

    /// <summary>Tolerated clock difference on signed requests, in seconds.</summary>
    public int ServiceClockSkewSeconds { get; set; } = 300;

    /// <summary>How long a used request nonce is remembered for replay protection.</summary>
    public int ServiceNonceRetentionSeconds { get; set; } = 600;
}
