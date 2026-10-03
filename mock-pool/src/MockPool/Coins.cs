namespace MockPool;

/// <summary>Which wire dialect a coin's listener speaks.</summary>
public enum CoinProtocol
{
    /// <summary>Classic Bitcoin-style Stratum V1: array params, subscribe, <c>[code,msg,tb]</c> errors.</summary>
    Classic,

    /// <summary>Pearl's named-params dialect: object params, no subscribe, object errors, 76-byte header.</summary>
    Pearl,
}

/// <summary>A coin the mock pool can mine. The code is the stable identifier (for example <c>prl</c>).</summary>
public sealed record CoinDefinition(string Code, string Name, CoinProtocol Protocol)
{
    /// <summary>Upper-case ticker used in job identifiers and log lines.</summary>
    public string Ticker => Code.ToUpperInvariant();
}

public static class Coins
{
    public static readonly CoinDefinition Pearl = new("prl", "Pearl", CoinProtocol.Pearl);

    // Quantus's pool-side dialect is not publicly specified; it runs classic V1 until we have one.
    public static readonly CoinDefinition Quantus = new("qtc", "Quantus", CoinProtocol.Classic);
}
