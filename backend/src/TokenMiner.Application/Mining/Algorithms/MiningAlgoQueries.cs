using MediatR;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;

namespace TokenMiner.Application.Mining.Algorithms;

public sealed record GetMiningAlgoQuery(Guid Id) : IRequest<MiningAlgoDto>;

internal sealed class GetMiningAlgoQueryHandler(IMiningAlgoRepository algorithms)
    : IRequestHandler<GetMiningAlgoQuery, MiningAlgoDto>
{
    public async Task<MiningAlgoDto> Handle(GetMiningAlgoQuery request, CancellationToken cancellationToken)
    {
        var algorithm = await algorithms.GetByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("Mining algorithm not found.");

        return algorithm.ToDto();
    }
}

public sealed record ListMiningAlgosQuery : IRequest<IReadOnlyList<MiningAlgoDto>>;

internal sealed class ListMiningAlgosQueryHandler(IMiningAlgoRepository algorithms)
    : IRequestHandler<ListMiningAlgosQuery, IReadOnlyList<MiningAlgoDto>>
{
    public async Task<IReadOnlyList<MiningAlgoDto>> Handle(
        ListMiningAlgosQuery request,
        CancellationToken cancellationToken)
    {
        var all = await algorithms.ListAsync(cancellationToken);

        // Highest priority first, which is the order the selection logic walks.
        return all.OrderByDescending(algorithm => algorithm.Priority)
            .Select(algorithm => algorithm.ToDto())
            .ToList();
    }
}
