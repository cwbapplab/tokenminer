using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class MiningSessionRepository(AppDbContext dbContext) : IMiningSessionRepository
{
    public Task<UserHardwareMiner?> GetByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.UserHardwareMiners.FirstOrDefaultAsync(session => session.Id == id, cancellationToken);

    public Task<UserHardwareMiner?> GetActiveByHardwareAsync(
        Guid userHardwareId,
        CancellationToken cancellationToken) =>
        dbContext.UserHardwareMiners.FirstOrDefaultAsync(
            session => session.UserHardwareId == userHardwareId
                && session.StoppedAt == null,
            cancellationToken);

    public Task<UserHardwareMiner?> GetActiveByWorkerIdentifierAsync(
        string workerIdentifier,
        CancellationToken cancellationToken) =>
        dbContext.UserHardwareMiners.FirstOrDefaultAsync(
            session => session.WorkerIdentifier == workerIdentifier
                && session.StoppedAt == null,
            cancellationToken);

    public async Task<IReadOnlyList<UserHardwareMiner>> ListActiveByHardwareIdsAsync(
        IEnumerable<Guid> hardwareIds,
        CancellationToken cancellationToken)
    {
        var ids = hardwareIds.Distinct().ToList();

        return await dbContext.UserHardwareMiners
            .Where(session => ids.Contains(session.UserHardwareId)
                && session.StoppedAt == null)
            .ToListAsync(cancellationToken);
    }

    public async Task<IReadOnlyList<UserHardwareMiner>> ListActiveAsync(CancellationToken cancellationToken) =>
        await dbContext.UserHardwareMiners
            .Where(session => session.StoppedAt == null)
            .ToListAsync(cancellationToken);

    public void Add(UserHardwareMiner session) => dbContext.UserHardwareMiners.Add(session);
}

internal sealed class MiningLogRepository(AppDbContext dbContext) : IMiningLogRepository
{
    public void Add(MiningLog log) => dbContext.MiningLogs.Add(log);
}
