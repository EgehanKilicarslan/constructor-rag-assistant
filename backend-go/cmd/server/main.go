package main

import (
	"fmt"
	"log"

	"github.com/EgehanKilicarslan/constructor-rag-assistant/backend-go/internal/api"
	"github.com/EgehanKilicarslan/constructor-rag-assistant/backend-go/internal/config"
	"github.com/EgehanKilicarslan/constructor-rag-assistant/backend-go/internal/rag"
)

func main() {
	// 1. Config
	cfg := config.LoadConfig()

	fmt.Printf("🚀 [Go] Starting Orchestrator... (Target: %s)\n", cfg.AIServiceAddr)

	// 2. Start RAG Client
	ragClient, err := rag.NewClient(cfg.AIServiceAddr, false)
	if err != nil {
		log.Fatalf("❌ Failed to connect to Python service: %v", err)
	}
	defer ragClient.Close()

	// 3. Setup API Handler and Router (Dependency Injection)
	handler := api.NewHandler(ragClient, cfg)
	r := api.SetupRouter(handler)

	// 4. Start Server
	addr := fmt.Sprintf(":%s", cfg.ApiServicePort)
	fmt.Printf("🌍 [Go] HTTP Server running on port %s\n", addr)
	if err := r.Run(addr); err != nil {
		log.Fatal(err)
	}
}
