using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddPoolPayouts : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<string>(
                name: "provider",
                table: "pools",
                type: "character varying(64)",
                maxLength: 64,
                nullable: false,
                defaultValue: "kryptex");

            migrationBuilder.CreateTable(
                name: "pool_payouts",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    pool_id = table.Column<Guid>(type: "uuid", nullable: false),
                    coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    wallet_address = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    amount = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    transaction_hash = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    requested_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true),
                    received_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true),
                    confirmations_count = table.Column<int>(type: "integer", nullable: false),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    raw_data = table.Column<string>(type: "jsonb", nullable: true),
                    redeemed_amount = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_pool_payouts", x => x.id);
                    table.ForeignKey(
                        name: "fk_pool_payouts_coins_coin_id",
                        column: x => x.coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_pool_payouts_pools_pool_id",
                        column: x => x.pool_id,
                        principalTable: "pools",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateIndex(
                name: "ix_pool_payouts_coin_id",
                table: "pool_payouts",
                column: "coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_pool_payouts_pool_id_transaction_hash",
                table: "pool_payouts",
                columns: new[] { "pool_id", "transaction_hash" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_pool_payouts_status_updated_at",
                table: "pool_payouts",
                columns: new[] { "status", "updated_at" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "pool_payouts");

            migrationBuilder.DropColumn(
                name: "provider",
                table: "pools");
        }
    }
}
