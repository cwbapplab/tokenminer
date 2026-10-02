using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata.Builders;
using TokenMiner.Domain.Users;
using TokenMiner.Domain.Users.Enums;

namespace TokenMiner.Infrastructure.Persistence.Configurations;

internal sealed class OtpCodeConfiguration : IEntityTypeConfiguration<OtpCode>
{
    public void Configure(EntityTypeBuilder<OtpCode> builder)
    {
        builder.HasKey(code => code.Id);

        builder.Property(code => code.Purpose)
            .IsRequired()
            .HasMaxLength(32)
            .HasConversion(
                purpose => purpose.ToDbValue(),
                value => OtpPurposeDbValues.FromDbValue(value));

        builder.Property(code => code.CodeHash)
            .IsRequired()
            .HasMaxLength(128);

        builder.Property(code => code.ExpiresAt).IsRequired();
        builder.Property(code => code.CreatedAt).IsRequired();

        builder.HasOne<User>()
            .WithMany()
            .HasForeignKey(code => code.UserId)
            .OnDelete(DeleteBehavior.Cascade);

        builder.HasIndex(code => new { code.UserId, code.Purpose, code.CreatedAt });
    }
}
