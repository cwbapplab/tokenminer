using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Domain.Users;

/// <summary>
/// A one-time code sent to the user's email. Only an HMAC of the code is stored, never the
/// code itself, and the number of verification attempts is capped.
/// </summary>
public sealed class OtpCode
{
    private OtpCode()
    {
    }

    public OtpCode(
        Guid id,
        Guid userId,
        OtpPurpose purpose,
        string codeHash,
        DateTimeOffset expiresAt,
        int maxAttempts,
        DateTimeOffset now)
    {
        Id = id;
        UserId = userId;
        Purpose = purpose;
        CodeHash = codeHash;
        ExpiresAt = expiresAt;
        MaxAttempts = maxAttempts;
        CreatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid UserId { get; private set; }

    public OtpPurpose Purpose { get; private set; }

    public string CodeHash { get; private set; } = null!;

    public DateTimeOffset ExpiresAt { get; private set; }

    public int Attempts { get; private set; }

    public int MaxAttempts { get; private set; }

    public DateTimeOffset? ConsumedAt { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public bool IsConsumed => ConsumedAt is not null;

    public bool IsExpired(DateTimeOffset now) => now >= ExpiresAt;

    public bool HasAttemptsRemaining => Attempts < MaxAttempts;

    public bool IsUsable(DateTimeOffset now) => !IsConsumed && !IsExpired(now) && HasAttemptsRemaining;

    public void RegisterFailedAttempt()
    {
        Attempts++;
    }

    public void Consume(DateTimeOffset now)
    {
        ConsumedAt = now;
    }
}
