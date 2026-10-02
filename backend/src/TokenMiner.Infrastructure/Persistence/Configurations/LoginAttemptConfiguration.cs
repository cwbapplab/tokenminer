using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Users;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class LoginAttemptConfiguration : IEntityTypeConfiguration<LoginAttempt>
{
    public void Configure(EntityTypeBuilder<LoginAttempt> builder)
    {
        builder.HasKey(attempt => attempt.Id);

        builder.Property(attempt => attempt.EmailNormalized)
            .IsRequired()
            .HasMaxLength(320);

        builder.Property(attempt => attempt.Ip).HasMaxLength(64);

        builder.Property(attempt => attempt.CreatedAt).IsRequired();

        builder.HasIndex(attempt => new { attempt.EmailNormalized, attempt.CreatedAt });
    }
}
