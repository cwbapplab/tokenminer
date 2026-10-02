using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Domain.Mining;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class MiningAlgoRepository(AppDbContext dbContext) : IMiningAlgoRepository
{
    public Task<MiningAlgo?> GetByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.MiningAlgos.FirstOrDefaultAsync(algorithm => algorithm.Id == id, cancellationToken);

    public Task<MiningAlgo?> GetByCodeAsync(string code, CancellationToken cancellationToken) =>
        dbContext.MiningAlgos.FirstOrDefaultAsync(algorithm => algorithm.Code == code, cancellationToken);

    public async Task<IReadOnlyList<MiningAlgo>> ListAsync(CancellationToken cancellationToken) =>
        await dbContext.MiningAlgos.ToListAsync(cancellationToken);

    public void Add(MiningAlgo algo) => dbContext.MiningAlgos.Add(algo);
}
