using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class UserHardwareRepository(AppDbContext dbContext) : IUserHardwareRepository
{
    public Task<UserHardware?> GetByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.UserHardware.FirstOrDefaultAsync(hardware => hardware.Id == id, cancellationToken);

    public async Task<IReadOnlyList<UserHardware>> ListByIdsAsync(
        IEnumerable<Guid> ids,
        CancellationToken cancellationToken)
    {
        var idList = ids.Distinct().ToList();

        return await dbContext.UserHardware
            .Where(hardware => idList.Contains(hardware.Id))
            .ToListAsync(cancellationToken);
    }

    public Task<UserHardware?> GetByUserAndHardwareIdAsync(
        Guid userId,
        Guid hardwareId,
        CancellationToken cancellationToken) =>
        dbContext.UserHardware.FirstOrDefaultAsync(
            hardware => hardware.UserId == userId && hardware.HardwareId == hardwareId,
            cancellationToken);

    public async Task<IReadOnlyList<UserHardware>> ListByUserAsync(Guid userId, CancellationToken cancellationToken) =>
        await dbContext.UserHardware
            .Where(hardware => hardware.UserId == userId)
            .ToListAsync(cancellationToken);

    public void Add(UserHardware hardware) => dbContext.UserHardware.Add(hardware);
}
