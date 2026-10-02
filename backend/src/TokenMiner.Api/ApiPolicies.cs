namespace TokenMiner.Api;

public static class AuthorizationPolicies
{
    public const string User = "UserPolicy";

    public const string Admin = "AdminPolicy";
}

public static class RateLimitPolicies
{
    /// <summary>Applied to the anonymous authentication endpoints.</summary>
    public const string Auth = "auth";
}
