using System.Text;
using System.Threading.RateLimiting;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.IdentityModel.Tokens;
using TokenMiner.Application.Authentication;
using TokenMiner.Domain.Users;

namespace TokenMiner.Api;

public static class AuthSetup
{
    public static IServiceCollection AddAuth(this IServiceCollection services, IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(configuration);

        var authOptions = configuration.GetSection(AuthOptions.SectionName).Get<AuthOptions>()
            ?? new AuthOptions();

        services
            .AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
            .AddJwtBearer(options =>
            {
                // Keep the compact claim names ("sub", "role") emitted at creation time.
                options.MapInboundClaims = false;

                options.TokenValidationParameters = new TokenValidationParameters
                {
                    ValidateIssuer = true,
                    ValidIssuer = authOptions.Jwt.Issuer,
                    ValidateAudience = true,
                    ValidAudience = authOptions.Jwt.Audience,
                    ValidateIssuerSigningKey = true,
                    IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(authOptions.Jwt.SigningKey)),
                    ValidateLifetime = true,
                    ClockSkew = TimeSpan.FromSeconds(30),
                    NameClaimType = AuthClaimTypes.Subject,
                    RoleClaimType = AuthClaimTypes.Role,
                };
            });

        services
            .AddAuthorizationBuilder()
            // Everything requires authentication unless it explicitly opts out.
            .SetFallbackPolicy(new AuthorizationPolicyBuilder().RequireAuthenticatedUser().Build())
            .AddPolicy(AuthorizationPolicies.User, policy => policy.RequireAuthenticatedUser())
            .AddPolicy(AuthorizationPolicies.Admin, policy => policy.RequireRole(Role.Admin));

        services.AddRateLimiter(options =>
        {
            options.RejectionStatusCode = StatusCodes.Status429TooManyRequests;

            options.AddPolicy(RateLimitPolicies.Auth, httpContext =>
                RateLimitPartition.GetFixedWindowLimiter(
                    partitionKey: httpContext.Connection.RemoteIpAddress?.ToString() ?? "unknown",
                    factory: _ => new FixedWindowRateLimiterOptions
                    {
                        PermitLimit = authOptions.RateLimit.PermitLimit,
                        Window = TimeSpan.FromSeconds(authOptions.RateLimit.WindowSeconds),
                        QueueLimit = 0,
                    }));
        });

        return services;
    }
}
