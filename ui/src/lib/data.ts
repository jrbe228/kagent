import type { Model, Tool } from "./types";

// TODO: Could also come from the backend
export const AVAILABLE_MODELS: Model[] = [
  { id: "gpt-4o", name: "GPT-4o" },
  { id: "gpt-4o-mini", name: "GPT-4o-mini" },
];


type Primitive = string | number | boolean;

export type InterfaceField = {
  key: string;
  type: 'string' | 'number' | 'boolean';
  required: boolean;
};

type InterfaceMetadata = {
  fields: Array<InterfaceField>;
};

// Helper function to create metadata for a config object
function createConfigMetadata<T>(config: {
  [K in keyof T]: T[K] extends Primitive ? T[K] : T[K] extends Primitive | undefined ? T[K] : never
}): InterfaceMetadata {
  const fields = Object.entries(config).map(([key, value]) => ({
    key,
    type: typeof value as 'string' | 'number' | 'boolean',
    required: value !== undefined
  }));

  return { fields };
}

interface PrometheusToolConfig {
  base_url: string;
  username?: string;
  password?: string;
}

const defaultPrometheusConfig: PrometheusToolConfig = {
  base_url: "http://localhost:9090/api/v1",
  username: undefined,
  password: undefined,
}

interface DocumentationConfig {
  docs_base_path?: string;
  docs_download_url?: string;
  openai_api_key?: string;
}


const defaultDocumentationConfig: DocumentationConfig = {
  docs_base_path: undefined,
  docs_download_url: undefined,
  openai_api_key: undefined,
}

// Used to render the UI
export const TOOL_CONFIGS = {
  'kagent.tools.prometheus': createConfigMetadata<PrometheusToolConfig>(defaultPrometheusConfig),
  'kagent.tools.docs': createConfigMetadata<DocumentationConfig>(defaultDocumentationConfig),
} as const;

export const getToolType = (provider: string): keyof typeof TOOL_CONFIGS | "unknown" => {
  if (provider.startsWith("kagent.tools.prometheus")) return "kagent.tools.prometheus";
  if (provider.startsWith("kagent.tools.docs")) return "kagent.tools.docs";
  return "unknown";
};


// TODO: This will come from the backend
export const TOOLS: Tool[] = [
  {
    provider: "kagent.tools.docs.QueryTool",
    label: "QueryTool",
    version: 1,
    component_version: 1,
    description: "Searches a vector database for relevant documentation of products such as Istio, Kubernetes, Prometheus",
    config: defaultDocumentationConfig,
  },
  {
    provider: "kagent.tools.prometheus.QueryTool",
    label: "QueryTool",
    version: 1,
    component_version: 1,
    description: "Tool for executing queries in Prometheus",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.QueryRangeTool",
    label: "QueryRangeTool",
    version: 1,
    component_version: 1,
    description: "Tool for executing range queries in Prometheus",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.SeriesQueryTool",
    label: "SeriesQueryTool",
    version: 1,
    component_version: 1,
    description: "Find series matching a metadata selector",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.LabelNamesTool",
    label: "LabelNamesTool",
    version: 1,
    component_version: 1,
    description: "Get all label names",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.LabelValuesTool",
    version: 1,
    label: "LabelValuesTool",
    component_version: 1,
    description: "Get values for a specific label",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.TargetsTool",
    version: 1,
    label: "TargetsTool",
    component_version: 1,
    description: "Provides information about all Prometheus scrape targets and their current state",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.RulesTool",
    version: 1,
    label: "RulesTool",
    component_version: 1,
    description: "Retrieves Prometheus alerting and recording rules",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.AlertsTool",
    version: 1,
    label: "AlertsTool",
    component_version: 1,
    description: "Retrieves active Prometheus alerts",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.TargetMetadataTool",
    version: 1,
    label: "TargetMetadataTool",
    component_version: 1,
    description: "Retrieves Prometheus target metadata",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.AlertmanagersTool",
    version: 1,
    label: "AlertmanagersTool",
    component_version: 1,
    description: "Retrieves Prometheus alertmanager discovery state",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.MetadataTool",
    version: 1,
    label: "MetadataTool",
    component_version: 1,
    description: "Retrieves Prometheus metric metadata",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.StatusConfigTool",
    version: 1,
    label: "StatusConfigTool",
    component_version: 1,
    description: "Retrieves Prometheus configuration",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.StatusFlagsTool",
    version: 1,
    label: "StatusFlagsTool",
    component_version: 1,
    description: "Retrieves Prometheus flag values",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.RuntimeInfoTool",
    version: 1,
    label: "RuntimeInfoTool",
    component_version: 1,
    description: "Retrieves Prometheus runtime information",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.BuildInfoTool",
    version: 1,
    label: "BuildInfoTool",
    component_version: 1,
    description: "Retrieves Prometheus build information",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.TSDBStatusTool",
    version: 1,
    label: "TSDBStatusTool",
    component_version: 1,
    description: "Retrieves Prometheus TSDB status",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.CreateSnapshotTool",
    version: 1,
    label: "CreateSnapshotTool",
    component_version: 1,
    description: "Creates Prometheus snapshots",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.DeleteSeriesTool",
    version: 1,
    label: "DeleteSeriesTool",
    component_version: 1,
    description: "Deletes Prometheus series data",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.CleanTombstonesTool",
    version: 1,
    label: "CleanTombstonesTool",
    component_version: 1,
    description: "Removes tombstones files created during delete operations",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.prometheus.WALReplayTool",
    version: 1,
    label: "WALReplayTool",
    component_version: 1,
    description: "Retrieves Prometheus Write-Ahead Log (WAL) replay status",
    config: defaultPrometheusConfig,
  },
  {
    provider: "kagent.tools.k8s.GetPods",
    label: "GetPods",
    version: 1,
    component_version: 1,
    description: "List pods in a namespace",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.GetServices",
    label: "GetServices",
    version: 1,
    component_version: 1,
    description: "List services in a namespace",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.GetPodLogs",
    label: "GetPodLogs",
    version: 1,
    component_version: 1,
    description: "Get logs from a pod",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.ApplyManifest",
    label: "ApplyManifest",
    version: 1,
    component_version: 1,
    description: "Apply a Kubernetes manifest",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.GetResources",
    label: "GetResources",
    version: 1,
    component_version: 1,
    description: "Get information about resources in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.AnnotateResource",
    label: "AnnotateResource",
    version: 1,
    component_version: 1,
    description: "Annotate a resource in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.CheckServiceConnectivity",
    label: "CheckServiceConnectivity",
    version: 1,
    component_version: 1,
    description: "Check connectivity to a service in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.CreateResource",
    label: "CreateResource",
    version: 1,
    component_version: 1,
    description: "Create a resource in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.DeleteResource",
    label: "DeleteResource",
    version: 1,
    component_version: 1,
    description: "Deletes a resource from Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.DescribeResource",
    label: "DescribeResource",
    version: 1,
    component_version: 1,
    description: "Describes a resource in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.GetAvailableAPIResources",
    label: "GetAvailableAPIResources",
    version: 1,
    component_version: 1,
    description: "Gets the supported API resources in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.GetClusterConfiguration",
    label: "GetClusterConfiguration",
    version: 1,
    component_version: 1,
    description: "Gets the Kubernetes cluster configuration",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.GetEvents",
    label: "GetEvents",
    version: 1,
    component_version: 1,
    description: "Gets the Kubernetes cluster events",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.LabelResource",
    label: "LabelResource",
    version: 1,
    component_version: 1,
    description: "Adds a label to a resource in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.RemoteLabel",
    label: "RemoteLabel",
    version: 1,
    component_version: 1,
    description: "Removes a label from a resource in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.PatchResource",
    label: "PatchResource",
    version: 1,
    component_version: 1,
    description: "Applies a patch to a resource in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.RemoveAnnotation",
    label: "RemoveAnnotation",
    version: 1,
    component_version: 1,
    description: "Removes an annotation from a resource in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.Rollout",
    label: "Rollout",
    version: 1,
    component_version: 1,
    description: "Removes an annotation from a resource in Kubernetes",
    config: {},
  },
  {
    provider: "kagent.tools.k8s.GetResourceYAML",
    label: "GetResourceYAML",
    version: 1,
    component_version: 1,
    description: "Performs a rollout on a resource in Kubernetes",
    config: {},
  },

];

export const isMcpTool = (tool: Tool) => tool.provider === 'autogen_ext.tools.mcp.StdioMcpToolAdapter';

// All MCP tools have the same label & description, so the actual tool name is stored in the config
export const getToolDisplayName = (tool: Tool) => isMcpTool(tool) && tool.config?.tool?.name ? tool.config.tool.name : tool.label;
export const getToolDescription = (tool: Tool) => isMcpTool(tool) && tool.config?.tool?.description ? tool.config.tool.description : tool.description;
export const getToolIdentifier = (tool: Tool): string => `${tool.provider}::${getToolDisplayName(tool)}`;
