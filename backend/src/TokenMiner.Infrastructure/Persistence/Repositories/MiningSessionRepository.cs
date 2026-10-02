using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;
using TokenMiner.Domain.Mining.Enums;

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
                && (session.Status == MiningSessionStatus.Running || session.Status == MiningSessionStatus.Paused),
            cancellationToken);

    public Task<UserHardwareMiner?> GetActiveByWorkerIdentifierAsync(
        string workerIdentifier,
        CancellationToken cancellationToken) =>
        dbContext.UserHardwareMiners.FirstOrDefaultAsync(
            session => session.WorkerIdentifier == workerIdentifier
                && (session.Status == MiningSessionStatus.Running || session.Status == MiningSessionStatus.Paused),
            cancellationToken);

    public async Task<IReadOnlyList<UserHardwareMiner>> ListRunningIdleSinceAsync(
        DateTimeOffset threshold,
        CancellationToken cancellationToken) =>
        await dbContext.UserHardwareMiners
            .Where(session => session.Status == MiningSessionStatus.Running
                && session.LastActivityAt < threshold)
            .ToListAsync(cancellationToken);

    public Task<int> CountByStatusAsync(MiningSessionStatus status, CancellationToken cancellationToken) =>
        dbContext.UserHardwareMiners.CountAsync(session => session.Status == status, cancellationToken);

    public void Add(UserHardwareMiner session) => dbContext.UserHardwareMiners.Add(session);
}

internal sealed class MiningLogRepository(AppDbContext dbContext) : IMiningLogRepository
{
    public void Add(MiningLog log) => dbContext.MiningLogs.Add(log);
}
