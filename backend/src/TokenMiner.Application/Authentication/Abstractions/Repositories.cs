using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Application.Authentication.Abstractions;

public interface IUserRepository
{
    Task<User?> GetByIdAsync(Guid id, CancellationToken cancellationToken);

    /// <summary>Newest accounts first, for the operator's user list.</summary>
    Task<IReadOnlyList<User>> ListAsync(int limit, CancellationToken cancellationToken);

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

    /// <summary>
    /// Atomically revokes a still-active token as part of a rotation.
    /// </summary>
    /// <returns>
    /// <c>true</c> when this caller claimed the rotation, <c>false</c> when the token was
    /// already revoked by a concurrent request. The write is issued as a conditional UPDATE so
    /// the check and the claim cannot be split apart.
    /// </returns>
    Task<bool> TryRotateAsync(
        Guid tokenId,
        string reason,
        DateTimeOffset now,
        Guid replacedByTokenId,
        CancellationToken cancellationToken);

    /// <summary>
    /// Revokes every still-active token descended from <paramref name="tokenId"/> by following
    /// the rotation chain, and returns how many were affected.
    /// </summary>
    /// <remarks>
    /// This is how a replay is contained: the chain is one session, so terminating it shuts the
    /// session down without touching the user's other devices. Walking the chain in the database
    /// keeps it to a single statement and needs no per-hop round trip.
    /// </remarks>
    Task<int> RevokeSessionChainAsync(
        Guid tokenId,
        string reason,
        DateTimeOffset now,
        CancellationToken cancellationToken);
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
