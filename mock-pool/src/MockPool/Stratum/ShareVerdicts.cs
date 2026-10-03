namespace MockPool.Stratum;

/// <summary>The outcome the pool reports for a submission.</summary>
public enum StratumVerdict
{
    /// <summary><c>result: true</c>.</summary>
    Accepted,

    /// <summary><c>result: false</c> with no error — a bare rejection.</summary>
    Rejected,

    /// <summary>Error 20 — other/unknown.</summary>
    Other,

    /// <summary>Error 21 — the job is not known (stale).</summary>
    JobNotFound,

    /// <summary>Error 22 — duplicate share.</summary>
    Duplicate,

    /// <summary>Error 23 — difficulty too low.</summary>
    LowDifficulty,

    /// <summary>Error 24 — the worker is not authorized.</summary>
    Unauthorized,

    /// <summary>Error 25 — the connection has not subscribed.</summary>
    NotSubscribed,

    /// <summary>Error 26 (Pearl) — the proof failed verification.</summary>
    InvalidProof,
}

public readonly record struct StratumError(int Code, string Message);

/// <summary>The standard Stratum error set, plus the protocol errors the pool can raise itself.</summary>
public static class StratumErrors
{
    public static readonly StratumError Other = new(20, "Other/unknown");
    public static readonly StratumError JobNotFound = new(21, "Job not found");
    public static readonly StratumError Duplicate = new(22, "Duplicate share");
    public static readonly StratumError LowDifficulty = new(23, "Low difficulty share");
    public static readonly StratumError Unauthorized = new(24, "Unauthorized worker");
    public static readonly StratumError NotSubscribed = new(25, "Not subscribed");
    public static readonly StratumError InvalidParams = new(20, "Invalid params");
    public static readonly StratumError UnknownMethod = new(20, "Unknown method");
    public static readonly StratumError InvalidProof = new(20, "Invalid proof");

    public static bool TryFor(StratumVerdict verdict, out StratumError error)
    {
        switch (verdict)
        {
            case StratumVerdict.Other:
                error = Other;
                return true;
            case StratumVerdict.JobNotFound:
                error = JobNotFound;
                return true;
            case StratumVerdict.Duplicate:
                error = Duplicate;
                return true;
            case StratumVerdict.LowDifficulty:
                error = LowDifficulty;
                return true;
            case StratumVerdict.Unauthorized:
                error = Unauthorized;
                return true;
            case StratumVerdict.NotSubscribed:
                error = NotSubscribed;
                return true;
            case StratumVerdict.InvalidProof:
                error = InvalidProof;
                return true;
            default:
                error = default;
                return false;
        }
    }
}

/// <summary>The error set of Pearl's named-params dialect (spec §3).</summary>
public static class PearlErrors
{
    public static readonly StratumError MethodNotSupported = new(20, "method not supported");
    public static readonly StratumError StaleJob = new(21, "stale job");
    public static readonly StratumError Duplicate = new(22, "duplicate share");
    public static readonly StratumError LowDifficulty = new(23, "low difficulty share");
    public static readonly StratumError WalletMissing = new(24, "wallet is missing");
    public static readonly StratumError InvalidWallet = new(25, "invalid wallet");
    public static readonly StratumError InvalidProof = new(26, "invalid proof");
    public static readonly StratumError Unauthorized = new(27, "unauthorized");

    public static bool TryFor(StratumVerdict verdict, out StratumError error)
    {
        switch (verdict)
        {
            case StratumVerdict.Other:
                error = MethodNotSupported;
                return true;
            case StratumVerdict.JobNotFound:
                error = StaleJob;
                return true;
            case StratumVerdict.Duplicate:
                error = Duplicate;
                return true;
            case StratumVerdict.LowDifficulty:
                error = LowDifficulty;
                return true;
            case StratumVerdict.InvalidProof:
                error = InvalidProof;
                return true;
            case StratumVerdict.Unauthorized:
            case StratumVerdict.NotSubscribed:
                error = Unauthorized;
                return true;
            default:
                error = default;
                return false;
        }
    }
}

/// <summary>
/// The mock's trigger convention. A submission's verdict is chosen from its tag — the first two hex
/// characters of the nonce for classic V1, or of the <c>plain_proof</c> for Pearl — falling back to
/// the job id, then to accepted.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><c>00</c> accepted, <c>20</c> other, <c>21</c> job not found, <c>22</c> duplicate,
/// <c>23</c> low difficulty, <c>24</c> unauthorized, <c>25</c> not subscribed, <c>26</c> invalid
/// proof, <c>ff</c> bare reject.</item>
/// <item>A tag is authoritative and is checked first, so it wins over the job id.</item>
/// <item>With no tag, a job that has been superseded by a newer block, or one whose id starts with
/// <c>stale</c> or <c>invalid</c>, reports job-not-found.</item>
/// <item>Any other submission is accepted.</item>
/// </list>
/// </remarks>
public static class ShareVerdicts
{
    public static StratumVerdict Resolve(string jobId, string tag) =>
        Resolve(jobId, tag, jobIsSuperseded: false);

    public static StratumVerdict Resolve(string jobId, string tag, bool jobIsSuperseded)
    {
        if (TryFromNonce(tag, out var verdict))
        {
            return verdict;
        }

        return jobIsSuperseded || IsStaleJob(jobId) ? StratumVerdict.JobNotFound : StratumVerdict.Accepted;
    }

    /// <summary>Reads a two-character tag (nonce prefix or proof prefix).</summary>
    public static bool TryFromNonce(string tag, out StratumVerdict verdict)
    {
        verdict = StratumVerdict.Accepted;
        if (tag.Length < 2)
        {
            return false;
        }

        switch (tag[..2].ToUpperInvariant())
        {
            case "00":
                verdict = StratumVerdict.Accepted;
                return true;
            case "20":
                verdict = StratumVerdict.Other;
                return true;
            case "21":
                verdict = StratumVerdict.JobNotFound;
                return true;
            case "22":
                verdict = StratumVerdict.Duplicate;
                return true;
            case "23":
                verdict = StratumVerdict.LowDifficulty;
                return true;
            case "24":
                verdict = StratumVerdict.Unauthorized;
                return true;
            case "25":
                verdict = StratumVerdict.NotSubscribed;
                return true;
            case "26":
                verdict = StratumVerdict.InvalidProof;
                return true;
            case "FF":
                verdict = StratumVerdict.Rejected;
                return true;
            default:
                return false;
        }
    }

    public static bool IsStaleJob(string jobId) =>
        jobId.StartsWith("stale", StringComparison.OrdinalIgnoreCase)
        || jobId.StartsWith("invalid", StringComparison.OrdinalIgnoreCase);
}
