namespace TokenMiner.Domain.Users;

/// <summary>
/// Single-use token backing the email validation link. Like refresh tokens, only the
/// hash is persisted.
/// </summary>
public sealed class EmailVerificationToken
{
    private EmailVerificationToken()
    {
    }

    public EmailVerificationToken(Guid id, Guid userId, string tokenHash, DateTimeOffset expiresAt, DateTimeOffset now)
    {
        Id = id;
        UserId = userId;
        TokenHash = tokenHash;
        ExpiresAt = expiresAt;
        CreatedAt = now;
    }

    public Guid Id { get; private set; }

    public Guid UserId { get; private set; }

    public string TokenHash { get; private set; } = null!;

    public DateTimeOffset ExpiresAt { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset? UsedAt { get; private set; }

    public bool IsUsable(DateTimeOffset now) => UsedAt is null && now < ExpiresAt;

    public void MarkUsed(DateTimeOffset now)
    {
        UsedAt ??= now;
    }
}
