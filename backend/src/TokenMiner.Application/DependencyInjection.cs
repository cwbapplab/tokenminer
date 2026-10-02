using FluentValidation;
using MediatR;
using Microsoft.Extensions.DependencyInjection;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Application.Authentication.Services;
using TokenMiner.Application.Common.Behaviors;

namespace TokenMiner.Application;

public static class DependencyInjection
{
    public static IServiceCollection AddApplication(this IServiceCollection services)
    {
        ArgumentNullException.ThrowIfNull(services);

        var assembly = typeof(DependencyInjection).Assembly;

        services.AddMediatR(configuration => configuration.RegisterServicesFromAssembly(assembly));
        services.AddValidatorsFromAssembly(assembly, includeInternalTypes: true);
        services.AddTransient(typeof(IPipelineBehavior<,>), typeof(ValidationBehavior<,>));

        services.AddSingleton(TimeProvider.System);
        services.AddScoped<IAuthTokenIssuer, AuthTokenIssuer>();

        return services;
    }
}
