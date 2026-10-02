using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Configuration.Abstractions;
using TokenMiner.Domain.Configuration;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

/// <summary>
/// Settings live in the database so an operator can retune the runtime without a deploy.
/// </summary>
internal sealed class SystemConfigurationStore(AppDbContext dbContext) : ISystemConfigurationStore
{
    public Task<string?> GetRawAsync(string key, CancellationToken cancellationToken) =>
        dbContext.SystemConfigurations
            .Where(setting => setting.Key == key)
            .Select(setting => setting.Value)
            .FirstOrDefaultAsync(cancellationToken);

    public async Task<T?> GetAsync<T>(string key, CancellationToken cancellationToken) =>
        SystemConfigurationStoreExtensions.Deserialize<T>(await GetRawAsync(key, cancellationToken));

    public async Task SetAsync(string key, string value, DateTimeOffset now, CancellationToken cancellationToken)
    {
        var existing = await dbContext.SystemConfigurations
            .FirstOrDefaultAsync(setting => setting.Key == key, cancellationToken);

        if (existing is null)
        {
            dbContext.SystemConfigurations.Add(new SystemConfiguration(key, value, now));
        }
        else
        {
            existing.Update(value, now);
        }

        // A setting that is only staged is not set, so this commits rather than deferring to the
        // caller's unit of work.
        await dbContext.SaveChangesAsync(cancellationToken);
    }

    public async Task<IReadOnlyList<SystemConfiguration>> ListAsync(CancellationToken cancellationToken) =>
        await dbContext.SystemConfigurations.ToListAsync(cancellationToken);
}
