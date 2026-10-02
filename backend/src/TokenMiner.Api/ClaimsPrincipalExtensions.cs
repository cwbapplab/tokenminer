using System.Security.Claims;
using TokenMiner.Application.Authentication;

namespace TokenMiner.Api;

internal static class ClaimsPrincipalExtensions
{
    /// <summary>Reads the authenticated user id from the access token's subject claim.</summary>
    public static Guid? GetUserId(this ClaimsPrincipal principal)
    {
        var subject = principal.FindFirstValue(AuthClaimTypes.Subject);

        return Guid.TryParse(subject, out var userId) ? userId : null;
    }
}
