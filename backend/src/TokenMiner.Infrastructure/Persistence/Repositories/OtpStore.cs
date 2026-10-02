using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class OtpStore(AppDbContext dbContext) : IOtpStore
{
    public void Add(OtpCode code) => dbContext.OtpCodes.Add(code);

    public Task<OtpCode?> GetLatestUsableAsync(
        Guid userId,
        OtpPurpose purpose,
        DateTimeOffset now,
        CancellationToken cancellationToken) =>
        dbContext.OtpCodes
            .Where(code => code.UserId == userId
                && code.Purpose == purpose
                && code.ConsumedAt == null
                && code.ExpiresAt > now
                && code.Attempts < code.MaxAttempts)
            .OrderByDescending(code => code.CreatedAt)
            .FirstOrDefaultAsync(cancellationToken);

    public Task<DateTimeOffset?> GetLastCreatedAtAsync(
        Guid userId,
        OtpPurpose purpose,
        CancellationToken cancellationToken) =>
        dbContext.OtpCodes
            .Where(code => code.UserId == userId && code.Purpose == purpose)
            .OrderByDescending(code => code.CreatedAt)
            .Select(code => (DateTimeOffset?)code.CreatedAt)
            .FirstOrDefaultAsync(cancellationToken);

    public async Task InvalidateOutstandingAsync(
        Guid userId,
        OtpPurpose purpose,
        DateTimeOffset now,
        CancellationToken cancellationToken)
    {
        var outstanding = await dbContext.OtpCodes
            .Where(code => code.UserId == userId && code.Purpose == purpose && code.ConsumedAt == null)
            .ToListAsync(cancellationToken);

        foreach (var code in outstanding)
        {
            code.Consume(now);
        }
    }
}
