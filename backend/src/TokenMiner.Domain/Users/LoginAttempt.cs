namespace TokenMiner.Domain.Users;

/// <summary>Audit row for a credential sign-in attempt; backs per-account lockout.</summary>
public sealed class LoginAttempt
{
    private LoginAttempt()
    {
    }

    public LoginAttempt(Guid id, string emailNormalized, string? ip, bool succeeded, DateTimeOffset now)
    {
        Id = id;
        EmailNormalized = emailNormalized;
        Ip = ip;
        Succeeded = succeeded;
        CreatedAt = now;
    }

    public Guid Id { get; private set; }

    public string EmailNormalized { get; private set; } = null!;

    public string? Ip { get; private set; }

    public bool Succeeded { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }
}
