using MediatR;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Application.Common.Exceptions;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Application.Authentication.Queries;

public sealed record GetCurrentUserQuery(Guid UserId) : IRequest<UserProfile>;

internal sealed class GetCurrentUserQueryHandler(
    IUserRepository users) : IRequestHandler<GetCurrentUserQuery, UserProfile>
{
    public async Task<UserProfile> Handle(GetCurrentUserQuery request, CancellationToken cancellationToken)
    {
        var user = await users.GetByIdAsync(request.UserId, cancellationToken)
            ?? throw new NotFoundException("User not found.");

        var roles = await users.GetRoleNamesAsync(user.Id, cancellationToken);

        return new UserProfile(
            user.Id,
            user.Email,
            user.DisplayName,
            user.Status.ToDbValue(),
            user.EmailConfirmedAt is not null,
            roles,
            user.CreatedAt);
    }
}
