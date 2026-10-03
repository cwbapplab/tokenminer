using Microsoft.EntityFrameworkCore;
using TokenMiner.Application.Authentication.Abstractions;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Repositories;

internal sealed class UserRepository(AppDbContext dbContext) : IUserRepository
{
    public Task<User?> GetByIdAsync(Guid id, CancellationToken cancellationToken) =>
        dbContext.Users.FirstOrDefaultAsync(user => user.Id == id, cancellationToken);

    public async Task<IReadOnlyList<User>> ListAsync(int limit, CancellationToken cancellationToken) =>
        await dbContext.Users
            .OrderByDescending(user => user.CreatedAt)
            .Take(limit)
            .ToListAsync(cancellationToken);

    public Task<User?> GetByNormalizedEmailAsync(string normalizedEmail, CancellationToken cancellationToken) =>
        dbContext.Users.FirstOrDefaultAsync(user => user.NormalizedEmail == normalizedEmail, cancellationToken);

    public Task<User?> GetByGoogleSubAsync(string googleSub, CancellationToken cancellationToken) =>
        dbContext.Users.FirstOrDefaultAsync(user => user.GoogleSub == googleSub, cancellationToken);

    public Task<bool> ExistsByNormalizedEmailAsync(string normalizedEmail, CancellationToken cancellationToken) =>
        dbContext.Users.AnyAsync(user => user.NormalizedEmail == normalizedEmail, cancellationToken);

    public void Add(User user) => dbContext.Users.Add(user);

    public async Task<IReadOnlyList<string>> GetRoleNamesAsync(Guid userId, CancellationToken cancellationToken) =>
        await (from userRole in dbContext.UserRoles
               join role in dbContext.Roles on userRole.RoleId equals role.Id
               where userRole.UserId == userId
               select role.Name).ToListAsync(cancellationToken);

    public Task<Role?> GetRoleByNameAsync(string name, CancellationToken cancellationToken) =>
        dbContext.Roles.FirstOrDefaultAsync(role => role.Name == name, cancellationToken);

    public void AddUserRole(UserRole userRole) => dbContext.UserRoles.Add(userRole);
}
