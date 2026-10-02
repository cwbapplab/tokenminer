using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class LoginAttemptStore(AppDbContext dbContext) : ILoginAttemptStore
{
    public void Add(LoginAttempt attempt) => dbContext.LoginAttempts.Add(attempt);

    public Task<int> CountRecentFailuresAsync(
        string normalizedEmail,
        DateTimeOffset since,
        CancellationToken cancellationToken) =>
        dbContext.LoginAttempts.CountAsync(
            attempt => attempt.EmailNormalized == normalizedEmail
                && !attempt.Succeeded
                && attempt.CreatedAt >= since,
            cancellationToken);
}
