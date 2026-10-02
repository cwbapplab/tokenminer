using System;
using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace TokenMiner.Infrastructure.Migrations
{
    /// <inheritdoc />
    public partial class AddConversionTreasury : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.CreateTable(
                name: "conversion_providers",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    name = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: false),
                    base_url = table.Column<string>(type: "character varying(512)", maxLength: 512, nullable: false),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    priority = table.Column<int>(type: "integer", nullable: false),
                    credential_ref = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    supported_features = table.Column<string>(type: "jsonb", nullable: true),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_conversion_providers", x => x.id);
                });

            migrationBuilder.CreateTable(
                name: "conversion_routes",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    conversion_provider_id = table.Column<Guid>(type: "uuid", nullable: false),
                    source_coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    destination_coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    source_network = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: true),
                    destination_network = table.Column<string>(type: "character varying(64)", maxLength: 64, nullable: true),
                    enabled = table.Column<bool>(type: "boolean", nullable: false),
                    priority = table.Column<int>(type: "integer", nullable: false),
                    minimum_amount = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_conversion_routes", x => x.id);
                    table.ForeignKey(
                        name: "fk_conversion_routes_coins_destination_coin_id",
                        column: x => x.destination_coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_conversion_routes_coins_source_coin_id",
                        column: x => x.source_coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_conversion_routes_conversion_providers_conversion_provider_",
                        column: x => x.conversion_provider_id,
                        principalTable: "conversion_providers",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Cascade);
                });

            migrationBuilder.CreateTable(
                name: "conversion_transactions",
                columns: table => new
                {
                    id = table.Column<Guid>(type: "uuid", nullable: false),
                    conversion_provider_id = table.Column<Guid>(type: "uuid", nullable: false),
                    source_coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    source_amount = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: false),
                    destination_coin_id = table.Column<Guid>(type: "uuid", nullable: false),
                    destination_amount = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    source_transaction_id = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    destination_transaction_id = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: true),
                    exchange_rate = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    fees = table.Column<decimal>(type: "numeric(28,10)", precision: 28, scale: 10, nullable: true),
                    status = table.Column<string>(type: "character varying(32)", maxLength: 32, nullable: false),
                    idempotency_key = table.Column<string>(type: "character varying(256)", maxLength: 256, nullable: false),
                    error = table.Column<string>(type: "character varying(1024)", maxLength: 1024, nullable: true),
                    created_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    updated_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: false),
                    completed_at = table.Column<DateTimeOffset>(type: "timestamp with time zone", nullable: true)
                },
                constraints: table =>
                {
                    table.PrimaryKey("pk_conversion_transactions", x => x.id);
                    table.ForeignKey(
                        name: "fk_conversion_transactions_coins_destination_coin_id",
                        column: x => x.destination_coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_conversion_transactions_coins_source_coin_id",
                        column: x => x.source_coin_id,
                        principalTable: "coins",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                    table.ForeignKey(
                        name: "fk_conversion_transactions_conversion_providers_conversion_pro",
                        column: x => x.conversion_provider_id,
                        principalTable: "conversion_providers",
                        principalColumn: "id",
                        onDelete: ReferentialAction.Restrict);
                });

            migrationBuilder.CreateIndex(
                name: "ix_conversion_providers_name",
                table: "conversion_providers",
                column: "name",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_conversion_routes_conversion_provider_id_source_coin_id_des",
                table: "conversion_routes",
                columns: new[] { "conversion_provider_id", "source_coin_id", "destination_coin_id" },
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_conversion_routes_destination_coin_id",
                table: "conversion_routes",
                column: "destination_coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_conversion_routes_source_coin_id_enabled_priority",
                table: "conversion_routes",
                columns: new[] { "source_coin_id", "enabled", "priority" });

            migrationBuilder.CreateIndex(
                name: "ix_conversion_transactions_conversion_provider_id",
                table: "conversion_transactions",
                column: "conversion_provider_id");

            migrationBuilder.CreateIndex(
                name: "ix_conversion_transactions_destination_coin_id",
                table: "conversion_transactions",
                column: "destination_coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_conversion_transactions_idempotency_key",
                table: "conversion_transactions",
                column: "idempotency_key",
                unique: true);

            migrationBuilder.CreateIndex(
                name: "ix_conversion_transactions_source_coin_id",
                table: "conversion_transactions",
                column: "source_coin_id");

            migrationBuilder.CreateIndex(
                name: "ix_conversion_transactions_status_updated_at",
                table: "conversion_transactions",
                columns: new[] { "status", "updated_at" });
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "conversion_routes");

            migrationBuilder.DropTable(
                name: "conversion_transactions");

            migrationBuilder.DropTable(
                name: "conversion_providers");
        }
    }
}
