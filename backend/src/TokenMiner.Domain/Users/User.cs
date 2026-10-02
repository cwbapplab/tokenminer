using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Domain.Users;

public sealed class User
{
    private User()
    {
    }

    public User(
        Guid id,
        string email,
        string normalizedEmail,
        string? passwordHash,
        string? displayName,
        DateTimeOffset now)
    {
        Id = id;
        Email = email;
        NormalizedEmail = normalizedEmail;
        PasswordHash = passwordHash;
        DisplayName = displayName;
        Status = UserStatus.PendingActivation;
        CreatedAt = now;
        UpdatedAt = now;
    }

    public Guid Id { get; private set; }

    public string Email { get; private set; } = null!;

    public string NormalizedEmail { get; private set; } = null!;

    /// <summary>Null for accounts created exclusively through an external provider.</summary>
    public string? PasswordHash { get; private set; }

    /// <summary>Google subject ("sub") claim, when the account is linked to Google.</summary>
    public string? GoogleSub { get; private set; }

    public string? DisplayName { get; private set; }

    public DateTimeOffset? EmailConfirmedAt { get; private set; }

    public UserStatus Status { get; private set; }

    public DateTimeOffset CreatedAt { get; private set; }

    public DateTimeOffset UpdatedAt { get; private set; }

    public bool IsActive => Status == UserStatus.Active;

    public void ConfirmEmail(DateTimeOffset now)
    {
        EmailConfirmedAt ??= now;
        UpdatedAt = now;
    }

    public void Activate(DateTimeOffset now)
    {
        Status = UserStatus.Active;
        UpdatedAt = now;
    }

    public void Suspend(DateTimeOffset now)
    {
        Status = UserStatus.Suspended;
        UpdatedAt = now;
    }

    public void SetPasswordHash(string passwordHash, DateTimeOffset now)
    {
        PasswordHash = passwordHash;
        UpdatedAt = now;
    }

    /// <summary>Links a Google subject. A different subject already on the account is never overwritten.</summary>
    public void LinkGoogle(string googleSub, DateTimeOffset now)
    {
        GoogleSub ??= googleSub;
        UpdatedAt = now;
    }

    public void UpdateProfile(string? displayName, DateTimeOffset now)
    {
        DisplayName = displayName;
        UpdatedAt = now;
    }
}
