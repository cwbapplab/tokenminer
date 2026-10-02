using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class EmailVerificationTokenStore(AppDbContext dbContext) : IEmailVerificationTokenStore
{
    public void Add(EmailVerificationToken token) => dbContext.EmailVerificationTokens.Add(token);

    public Task<EmailVerificationToken?> GetByHashAsync(string tokenHash, CancellationToken cancellationToken) =>
        dbContext.EmailVerificationTokens
            .FirstOrDefaultAsync(token => token.TokenHash == tokenHash, cancellationToken);
}
