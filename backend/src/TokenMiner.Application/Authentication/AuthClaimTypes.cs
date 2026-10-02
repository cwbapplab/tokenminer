namespace TokenMiner.Application.Authentication;

/// <summary>
/// Claim names emitted in access tokens. Short names are used deliberately so the token
/// needs no inbound claim-type remapping when validated.
/// </summary>
public static class AuthClaimTypes
{
    public const string Subject = "sub";

    public const string Email = "email";

    public const string JwtId = "jti";

    public const string Role = "role";
}
