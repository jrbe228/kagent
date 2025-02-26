package cli

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/abiosoft/ishell/v2"
	"github.com/kagent-dev/kagent/go/autogen/api"
	"github.com/kagent-dev/kagent/go/cli/internal/config"
)

func GetCmd(c *ishell.Context) {
	cfg, err := config.Get()
	if err != nil {
		fmt.Printf("Failed to get config: %v\n", err)
		return
	}

	client := api.NewClient(cfg.APIURL, cfg.WSURL)

	resourceType := ""
	resourceName := ""
	for idx, arg := range c.Args {
		if idx == 0 {
			resourceType = arg
		} else if idx == 1 {
			resourceName = arg
		}
	}

	// Lowercase the resource type and remove the plural "s" if it exists
	resourceType = strings.TrimSuffix(strings.ToLower(resourceType), "s")
	switch resourceType {
	case "run":
		if resourceName == "" {
			runList, err := client.ListRuns(cfg.UserID)
			if err != nil {
				c.Printf("Failed to get runs: %v\n", err)
				return
			}

			if err := printRuns(runList); err != nil {
				c.Printf("Failed to print runs: %v\n", err)
				return
			}
		} else {
			run, err := client.GetRun(resourceName)
			if err != nil {
				c.Printf("Failed to get run %s: %v\n", resourceName, err)
				return
			}
			c.Printf("Run %s: %v\n", resourceName, run)
		}
	case "agent":
		if resourceName == "" {
			agentList, err := client.ListTeams(cfg.UserID)
			if err != nil {
				c.Printf("Failed to get agents: %v\n", err)
				return
			}
			if err := printTeams(agentList); err != nil {
				c.Printf("Failed to print agents: %v\n", err)
				return
			}
		} else {
			agent, err := client.GetTeam(resourceName, cfg.UserID)
			if err != nil {
				c.Printf("Failed to get agent %s: %v\n", resourceName, err)
				return
			}
			c.Printf("Agent %s: %v\n", resourceName, agent)
		}
	case "session":
		c.Printf("Getting session %s", resourceName)
	default:
		c.Printf("Unknown resource type: %s", resourceType)
	}
}

func printRuns(runs []api.Run) error {
	headers := []string{"ID", "CONTENT", "MESSAGES", "STATUS", "CREATED"}
	rows := make([][]string, len(runs))
	for i, run := range runs {
		// Truncate task content to first 10 characters if possible
		content := run.Task.Content
		if len(content) > 10 {
			content = content[:10] + "..."
		}

		rows[i] = []string{
			run.ID,
			content,
			strconv.Itoa(len(run.Messages)),
			run.Status,
			run.CreatedAt,
		}
	}

	return printOutput(runs, headers, rows)
}

func printTeams(teams []api.Team) error {
	// Prepare table data
	headers := []string{"NAME", "CREATED"}
	rows := make([][]string, len(teams))
	for i, team := range teams {
		rows[i] = []string{
			team.Component.Label,
			team.CreatedAt,
		}
	}

	return printOutput(teams, headers, rows)
}
