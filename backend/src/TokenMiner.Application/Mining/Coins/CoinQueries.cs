using MediatR;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Application.Mining.Abstractions;
using TokenMiner.Application.Mining.Models;

namespace TokenMiner.Application.Mining.Coins;

public sealed record GetCoinQuery(Guid Id) : IRequest<CoinDto>;

internal sealed class GetCoinQueryHandler(ICoinRepository coins) : IRequestHandler<GetCoinQuery, CoinDto>
{
    public async Task<CoinDto> Handle(GetCoinQuery request, CancellationToken cancellationToken)
    {
        var coin = await coins.GetByIdAsync(request.Id, cancellationToken)
            ?? throw new NotFoundException("Coin not found.");

        return coin.ToDto();
    }
}

public sealed record ListCoinsQuery : IRequest<IReadOnlyList<CoinDto>>;

internal sealed class ListCoinsQueryHandler(ICoinRepository coins)
    : IRequestHandler<ListCoinsQuery, IReadOnlyList<CoinDto>>
{
    public async Task<IReadOnlyList<CoinDto>> Handle(
        ListCoinsQuery request,
        CancellationToken cancellationToken)
    {
        var all = await coins.ListAsync(cancellationToken);

        return all.Select(coin => coin.ToDto()).ToList();
    }
}
