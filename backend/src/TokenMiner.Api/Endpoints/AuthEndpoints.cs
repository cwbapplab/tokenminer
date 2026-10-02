using System.Security.Claims;
using MediatR;
using TokenMiner.Application.Authentication;
using TokenMiner.Application.Authentication.Commands;
using TokenMiner.Application.Authentication.Models;
using TokenMiner.Application.Authentication.Queries;
using TokenMiner.Contracts.Auth;

namespace TokenMiner.Api.Endpoints;

public static class AuthEndpoints
{
    public static IEndpointRouteBuilder MapAuthEndpoints(this IEndpointRouteBuilder app)
    {
        ArgumentNullException.ThrowIfNull(app);

        var anonymous = app.MapGroup("/api/auth")
            .WithTags("Auth")
            .RequireRateLimiting(RateLimitPolicies.Auth)
            .AllowAnonymous();

        anonymous.MapPost("/register", RegisterAsync)
            .WithName("Register")
            .Produces<PendingActivationResponse>(StatusCodes.Status202Accepted)
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status409Conflict);

        anonymous.MapPost("/verify-email", VerifyEmailAsync)
            .WithName("VerifyEmail")
            .Produces(StatusCodes.Status200OK)
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        anonymous.MapPost("/otp/request", RequestOtpAsync)
            .WithName("RequestOtp")
            .Produces(StatusCodes.Status202Accepted)
            .ProducesProblem(StatusCodes.Status429TooManyRequests);

        anonymous.MapPost("/otp/verify", VerifyOtpAsync)
            .WithName("VerifyOtp")
            .Produces<AuthTokensResponse>()
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        anonymous.MapPost("/google", GoogleSignInAsync)
            .WithName("GoogleSignIn")
            .Produces<AuthTokensResponse>()
            .Produces<PendingActivationResponse>(StatusCodes.Status202Accepted)
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        anonymous.MapPost("/login", LoginAsync)
            .WithName("Login")
            .Produces<AuthTokensResponse>()
            .ProducesProblem(StatusCodes.Status401Unauthorized)
            .ProducesProblem(StatusCodes.Status403Forbidden)
            .ProducesProblem(StatusCodes.Status429TooManyRequests);

        anonymous.MapPost("/refresh", RefreshAsync)
            .WithName("RefreshToken")
            .Produces<AuthTokensResponse>()
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        anonymous.MapPost("/logout", LogoutAsync)
            .WithName("Logout")
            .Produces(StatusCodes.Status204NoContent);

        app.MapGet("/api/auth/me", GetCurrentUserAsync)
            .WithTags("Auth")
            .WithName("GetCurrentUser")
            .RequireAuthorization(AuthorizationPolicies.User)
            .Produces<UserProfileResponse>()
            .ProducesProblem(StatusCodes.Status401Unauthorized);

        return app;
    }

    private static async Task<IResult> RegisterAsync(
        RegisterRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var result = await sender.Send(
            new RegisterCommand(request.Email, request.Password, request.DisplayName),
            cancellationToken);

        return Results.Json(
            new PendingActivationResponse(result.RequiresActivation),
            statusCode: StatusCodes.Status202Accepted);
    }

    private static async Task<IResult> VerifyEmailAsync(
        VerifyEmailRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        await sender.Send(new VerifyEmailCommand(request.Token), cancellationToken);
        return Results.Ok();
    }

    private static async Task<IResult> RequestOtpAsync(
        RequestOtpRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        await sender.Send(new RequestOtpCommand(request.Email), cancellationToken);
        return Results.StatusCode(StatusCodes.Status202Accepted);
    }

    private static async Task<IResult> VerifyOtpAsync(
        VerifyOtpRequest request,
        HttpContext httpContext,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var tokens = await sender.Send(
            new VerifyOtpCommand(request.Email, request.Code, GetIp(httpContext), GetUserAgent(httpContext)),
            cancellationToken);

        return Results.Ok(ToResponse(tokens));
    }

    private static async Task<IResult> GoogleSignInAsync(
        GoogleSignInRequest request,
        HttpContext httpContext,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var result = await sender.Send(
            new GoogleSignInCommand(request.IdToken, GetIp(httpContext), GetUserAgent(httpContext)),
            cancellationToken);

        return result.RequiresActivation
            ? Results.Json(new PendingActivationResponse(true), statusCode: StatusCodes.Status202Accepted)
            : Results.Ok(ToResponse(result.Tokens!));
    }

    private static async Task<IResult> LoginAsync(
        LoginRequest request,
        HttpContext httpContext,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var tokens = await sender.Send(
            new LoginCommand(request.Email, request.Password, GetIp(httpContext), GetUserAgent(httpContext)),
            cancellationToken);

        return Results.Ok(ToResponse(tokens));
    }

    private static async Task<IResult> RefreshAsync(
        RefreshTokenRequest request,
        HttpContext httpContext,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var tokens = await sender.Send(
            new RefreshTokenCommand(request.RefreshToken, GetIp(httpContext), GetUserAgent(httpContext)),
            cancellationToken);

        return Results.Ok(ToResponse(tokens));
    }

    private static async Task<IResult> LogoutAsync(
        LogoutRequest request,
        ISender sender,
        CancellationToken cancellationToken)
    {
        await sender.Send(new LogoutCommand(request.RefreshToken), cancellationToken);
        return Results.NoContent();
    }

    private static async Task<IResult> GetCurrentUserAsync(
        HttpContext httpContext,
        ISender sender,
        CancellationToken cancellationToken)
    {
        var subject = httpContext.User.FindFirstValue(AuthClaimTypes.Subject);
        if (!Guid.TryParse(subject, out var userId))
        {
            return Results.Unauthorized();
        }

        var profile = await sender.Send(new GetCurrentUserQuery(userId), cancellationToken);

        return Results.Ok(new UserProfileResponse(
            profile.Id,
            profile.Email,
            profile.DisplayName,
            profile.Status,
            profile.EmailConfirmed,
            profile.Roles,
            profile.CreatedAt));
    }

    private static AuthTokensResponse ToResponse(AuthTokens tokens) =>
        new(tokens.AccessToken, tokens.RefreshToken, tokens.ExpiresIn);

    private static string? GetIp(HttpContext httpContext) =>
        httpContext.Connection.RemoteIpAddress?.ToString();

    private static string? GetUserAgent(HttpContext httpContext) =>
        httpContext.Request.Headers["User-Agent"].ToString();
}
