using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using TokenMiner.Application.Authentication;
using TokenMiner.Domain.Users;
using TokenMiner.Infrastructure.Persistence;

namespace TokenMiner.Infrastructure.Authentication;

/// <summary>
/// Grants the admin role to a configured set of existing accounts at startup, so the
/// administrative endpoints are reachable without hand-editing the database. No-op when
/// <see cref="AuthOptions.BootstrapAdminEmails"/> is empty.
/// </summary>
public sealed class AdminRoleBootstrapper(
    AppDbContext dbContext,
    AuthOptions options,
    ILogger<AdminRoleBootstrapper> logger)
{
    public async Task EnsureAsync(CancellationToken cancellationToken)
    {
        foreach (var email in options.BootstrapAdminEmails)
        {
            if (string.IsNullOrWhiteSpace(email))
            {
                continue;
            }

            var normalizedEmail = EmailNormalizer.Normalize(email);

            var user = await dbContext.Users
                .FirstOrDefaultAsync(candidate => candidate.NormalizedEmail == normalizedEmail, cancellationToken);

            if (user is null)
            {
                logger.LogWarning("Bootstrap admin {Email} has no account yet; skipping.", email);
                continue;
            }

            var alreadyAdmin = await dbContext.UserRoles
                .AnyAsync(userRole => userRole.UserId == user.Id && userRole.RoleId == RoleIds.Admin, cancellationToken);

            if (alreadyAdmin)
            {
                continue;
            }

            dbContext.UserRoles.Add(new UserRole(user.Id, RoleIds.Admin));
            await dbContext.SaveChangesAsync(cancellationToken);

            logger.LogInformation("Granted the admin role to {Email}.", email);
        }
    }
}
