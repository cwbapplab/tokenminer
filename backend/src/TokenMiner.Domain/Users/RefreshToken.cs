namespace TokenMiner.Domain.Users;

/// <summary>
/// A refresh token. Only the SHA-256 hash of the opaque token value is persisted, so a
/// database leak does not expose usable tokens. Rotation is tracked through
/// <see cref="ReplacedByTokenId"/> so reuse of an already-rotated token can be detected.
/// </summary>
public sealed class RefreshToken
{
    private RefreshToken()
    {
    }

    public RefreshToken(
        Guid id,
        Guid userId,
        string tokenHash,
        DateTimeOffset expiresAt,
        DateTimeOffset now,
        string? createdByIp,
        string? userAgent)
    {
        Id = id;
        UserId = userId;
        TokenHash = tokenHash;
        ExpiresAt = expiresAt;
        CreatedAt = now;
        CreatedByIp = createdByIp;
        UserAgent = userAgent;
    }

    public Guid Id { get; private set; }

    public Guid UserId { get; private set; }

    public string TokenHash { get; private set; } = null!;

    public DateTimeOffset ExpiresAt { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset? RevokedAt { get; private set; }

    /// <summary>Token that superseded this one during rotation.</summary>
    public Guid? ReplacedByTokenId { get; private set; }

    public string? CreatedByIp { get; private set; }

    public string? UserAgent { get; private set; }

    public string? RevokeReason { get; private set; }

    public bool IsRevoked => RevokedAt is not null;

    public bool IsExpired(DateTimeOffset now) => now >= ExpiresAt;

    public bool IsActive(DateTimeOffset now) => !IsRevoked && !IsExpired(now);

    public void Revoke(string reason, DateTimeOffset now, Guid? replacedByTokenId = null)
    {
        if (IsRevoked)
        {
            return;
        }

        RevokedAt = now;
        RevokeReason = reason;
        ReplacedByTokenId = replacedByTokenId;
    }
}
