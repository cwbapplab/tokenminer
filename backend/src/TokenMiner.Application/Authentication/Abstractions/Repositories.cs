using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Application.Authentication.Abstractions;

public interface IUserRepository
{
    Task<User?> GetByIdAsync(Guid id, CancellationToken cancellationToken);

    Task<User?> GetByNormalizedEmailAsync(string normalizedEmail, CancellationToken cancellationToken);

    Task<User?> GetByGoogleSubAsync(string googleSub, CancellationToken cancellationToken);

    Task<bool> ExistsByNormalizedEmailAsync(string normalizedEmail, CancellationToken cancellationToken);

    void Add(User user);

    Task<IReadOnlyList<string>> GetRoleNamesAsync(Guid userId, CancellationToken cancellationToken);

    Task<Role?> GetRoleByNameAsync(string name, CancellationToken cancellationToken);

    void AddUserRole(UserRole userRole);
}

public interface IRefreshTokenStore
{
    void Add(RefreshToken token);

    Task<RefreshToken?> GetByHashAsync(string tokenHash, CancellationToken cancellationToken);

    /// <summary>Every token for the user that is neither revoked nor expired.</summary>
    Task<IReadOnlyList<RefreshToken>> GetActiveByUserAsync(Guid userId, CancellationToken cancellationToken);
}

public interface IOtpStore
{
    void Add(OtpCode code);

    Task<OtpCode?> GetLatestUsableAsync(
        Guid userId,
        OtpPurpose purpose,
        DateTimeOffset now,
        CancellationToken cancellationToken);

    /// <summary>Creation time of the most recent code for this user/purpose, for resend throttling.</summary>
    Task<DateTimeOffset?> GetLastCreatedAtAsync(
        Guid userId,
        OtpPurpose purpose,
        CancellationToken cancellationToken);

    /// <summary>Consumes any outstanding codes so only the newest one can be used.</summary>
    Task InvalidateOutstandingAsync(Guid userId, OtpPurpose purpose, DateTimeOffset now, CancellationToken cancellationToken);
}

public interface IEmailVerificationTokenStore
{
    void Add(EmailVerificationToken token);

    Task<EmailVerificationToken?> GetByHashAsync(string tokenHash, CancellationToken cancellationToken);
}

public interface ILoginAttemptStore
{
    void Add(LoginAttempt attempt);

    Task<int> CountRecentFailuresAsync(string normalizedEmail, DateTimeOffset since, CancellationToken cancellationToken);
}
