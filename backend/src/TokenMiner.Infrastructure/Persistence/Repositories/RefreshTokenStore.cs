using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class RefreshTokenStore(AppDbContext dbContext) : IRefreshTokenStore
{
    public void Add(RefreshToken token) => dbContext.RefreshTokens.Add(token);

    public Task<RefreshToken?> GetByHashAsync(string tokenHash, CancellationToken cancellationToken) =>
        dbContext.RefreshTokens.FirstOrDefaultAsync(token => token.TokenHash == tokenHash, cancellationToken);

    public async Task<IReadOnlyList<RefreshToken>> GetActiveByUserAsync(Guid userId, CancellationToken cancellationToken) =>
        await dbContext.RefreshTokens
            .Where(token => token.UserId == userId && token.RevokedAt == null)
            .ToListAsync(cancellationToken);
}
