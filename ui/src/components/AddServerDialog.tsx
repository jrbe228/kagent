import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Terminal, Globe, Loader2, ChevronDown, ChevronUp, PlusCircle, Trash2, Code } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Component, SseMcpServerConfig, StdioMcpServerConfig, ToolServerConfig } from "@/types/datamodel";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

interface AddServerDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onAddServer: (serverConfig: Component<ToolServerConfig>) => void;
}

interface ArgPair {
  value: string;
}

interface EnvPair {
  key: string;
  value: string;
}

export function AddServerDialog({ open, onOpenChange, onAddServer }: AddServerDialogProps) {
  const [activeTab, setActiveTab] = useState<"command" | "url">("command");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showAdvancedCommand, setShowAdvancedCommand] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [serverName, setServerName] = useState("");
  const [autoGeneratedName, setAutoGeneratedName] = useState("");

  // Command structure fields
  const [commandType, setCommandType] = useState("npx");
  const [commandPrefix, setCommandPrefix] = useState("");
  const [packageName, setPackageName] = useState("");

  // Arguments in simplified format (single value)
  const [argPairs, setArgPairs] = useState<ArgPair[]>([{ value: "" }]);

  // Environment variables as key-value pairs
  const [envPairs, setEnvPairs] = useState<EnvPair[]>([{ key: "", value: "" }]);

  // Command preview
  const [commandPreview, setCommandPreview] = useState("");

  // StdioServer parameters
  const [stderr, setStderr] = useState("");
  const [cwd, setCwd] = useState("");

  // SseServer parameters
  const [url, setUrl] = useState("");
  const [headers, setHeaders] = useState("");
  const [timeout, setTimeout] = useState("5");
  const [sseReadTimeout, setSseReadTimeout] = useState("300");

  // Clean up package name for server name
  const cleanPackageName = (pkgName: string): string => {
    // Remove organization prefix if exists (e.g., @org/package-name -> package-name)
    let cleaned = pkgName.trim().replace(/^@[\w-]+\//, "");

    // Remove common prefixes like "mcp-" if followed by a descriptive name
    if (cleaned.startsWith("mcp-") && cleaned.length > 4) {
      cleaned = cleaned.substring(4);
    }

    // Handle hyphenated names better
    cleaned = cleaned
      .split("-")
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
      .join(" ");

    // Add "Server" suffix if not already present
    if (!cleaned.toLowerCase().includes("server")) {
      cleaned += " Server";
    }

    return cleaned;
  };

  // Auto-generate server name when package name or URL changes
  useEffect(() => {
    let generatedName = "";

    if (activeTab === "command" && packageName.trim()) {
      generatedName = cleanPackageName(packageName.trim());
    } else if (activeTab === "url" && url.trim()) {
      try {
        const urlObj = new URL(url.trim());
        generatedName = `${urlObj.hostname} Server`;
      } catch {
        // If URL is invalid, just use what's available
        generatedName = `Remote Server`;
      }
    }

    setAutoGeneratedName(generatedName);
    setServerName(generatedName);
  }, [activeTab, packageName, url]);

  // Update command preview whenever inputs change
  useEffect(() => {
    if (activeTab === "command") {
      let preview = commandType;

      // Add command prefix if present
      if (commandPrefix.trim()) {
        preview += " " + commandPrefix.trim();
      }

      // Add package name if present
      if (packageName.trim()) {
        preview += " " + packageName.trim();
      }

      // Add all non-empty arguments
      argPairs.forEach((arg) => {
        if (arg.value.trim()) {
          preview += " " + arg.value.trim();
        }
      });

      setCommandPreview(preview);
    }
  }, [activeTab, commandType, commandPrefix, packageName, argPairs]);

  const addArgPair = () => {
    setArgPairs([...argPairs, { value: "" }]);
  };

  const removeArgPair = (index: number) => {
    setArgPairs(argPairs.filter((_, i) => i !== index));
  };

  const updateArgPair = (index: number, newValue: string) => {
    const updatedPairs = [...argPairs];
    updatedPairs[index].value = newValue;
    setArgPairs(updatedPairs);
  };

  const addEnvPair = () => {
    setEnvPairs([...envPairs, { key: "", value: "" }]);
  };

  const removeEnvPair = (index: number) => {
    setEnvPairs(envPairs.filter((_, i) => i !== index));
  };

  const updateEnvPair = (index: number, field: "key" | "value", newValue: string) => {
    const updatedPairs = [...envPairs];
    updatedPairs[index][field] = newValue;
    setEnvPairs(updatedPairs);
  };

  const formatArgs = (): string[] => {
    const args: string[] = [];

    // Add command prefix to the args array if present
    if (commandPrefix.trim()) {
      // Split on whitespace to handle cases where command prefix has multiple parts
      args.push(...commandPrefix.trim().split(/\s+/));
    }

    // Add package name if present
    if (packageName.trim()) {
      args.push(packageName.trim());
    }

    // Add all additional arguments
    argPairs.filter((arg) => arg.value.trim() !== "").forEach((arg) => args.push(arg.value.trim()));

    return args;
  };

  const formatEnvVars = (): Record<string, string> => {
    const envVars: Record<string, string> = {};
    envPairs.forEach((pair) => {
      if (pair.key.trim() && pair.value.trim()) {
        envVars[pair.key.trim()] = pair.value.trim();
      }
    });
    return envVars;
  };

  const handleSubmit = () => {
    if (activeTab === "command" && !packageName.trim()) {
      setError("Package name is required");
      return;
    }

    if (activeTab === "url" && !url.trim()) {
      setError("URL is required");
      return;
    }

    setIsSubmitting(true);

    let params: StdioMcpServerConfig | SseMcpServerConfig;
    if (activeTab === "command") {
      // Create StdioServerParameters
      params = {
        command: commandType,
        args: formatArgs(),
      };

      // Add environment variables if any exist
      const formattedEnv = formatEnvVars();
      if (Object.keys(formattedEnv).length > 0) {
        params.env = formattedEnv;
      }

      // Add optional parameters if they exist
      if (stderr.trim()) {
        params.stderr = stderr;
      }

      if (cwd.trim()) {
        params.cwd = cwd;
      }
    } else {
      params = {
        url: url.trim(),
      };

      // Add optional parameters if they exist
      if (headers.trim()) {
        try {
          params.headers = JSON.parse(headers);
          // eslint-disable-next-line @typescript-eslint/no-unused-vars
        } catch (e) {
          setError("Headers must be valid JSON");
          setIsSubmitting(false);
          return;
        }
      }

      if (timeout.trim()) {
        params.timeout = Number(timeout);
      }

      if (sseReadTimeout.trim()) {
        params.sse_read_timeout = Number(sseReadTimeout);
      }
    }

    const newServer: Component<ToolServerConfig> = {
      provider: "command" in params ? "kagent.tool_servers.StdioMcpToolServer" : "kagent.tool_servers.SseMcpToolServer",
      component_type: "tool_server",
      version: 1,
      component_version: 1,
      label: serverName || autoGeneratedName,
      config: params,
    };

    onAddServer(newServer);
    resetForm();
    setIsSubmitting(false);
  };

  const resetForm = () => {
    setCommandType("npx");
    setCommandPrefix("");
    setPackageName("");
    setArgPairs([{ value: "" }]);
    setEnvPairs([{ key: "", value: "" }]);
    setServerName("");
    setStderr("");
    setCwd("");
    setUrl("");
    setHeaders("");
    setTimeout("5");
    setSseReadTimeout("300");
    setError(null);
    setShowAdvanced(false);
    setShowAdvancedCommand(false);
    setCommandPreview("");
    setAutoGeneratedName("");
  };

  const handleClose = () => {
    resetForm();
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Add Tool Server</DialogTitle>
        </DialogHeader>

        <div className="py-4">
          {error && <div className="mb-4 p-2 bg-red-50 border border-red-200 text-red-700 rounded text-sm">{error}</div>}

          <div className="space-y-4">
            <Tabs defaultValue="command" value={activeTab} onValueChange={(v) => setActiveTab(v as "command" | "url")}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="command" className="flex items-center gap-2">
                  <Terminal className="h-4 w-4" />
                  Command
                </TabsTrigger>
                <TabsTrigger value="url" className="flex items-center gap-2">
                  <Globe className="h-4 w-4" />
                  URL
                </TabsTrigger>
              </TabsList>

              <TabsContent value="command" className="pt-4 space-y-4">
                {/* Command Preview Box */}
                <div className="p-3 bg-gray-50 border rounded-md font-mono text-sm">
                  <div className="flex items-center gap-2 mb-1 text-gray-500">
                    <Code className="h-4 w-4" />
                    <span>Command Preview:</span>
                  </div>
                  <div className="overflow-x-auto">
                    <div className="whitespace-pre-wrap break-all">{commandPreview || "<command will appear here>"}</div>
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-4">
                  <div className="space-y-2">
                    <Label>Command Executor</Label>
                    <Select value={commandType} onValueChange={setCommandType}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select command" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="npx">npx</SelectItem>
                        <SelectItem value="uvx">uvx</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">Select the command executor (e.g., npx or uvx)</p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between mb-1">
                    <Label htmlFor="package-name">Package Name</Label>
                    <Button variant="ghost" size="sm" onClick={() => setShowAdvancedCommand(!showAdvancedCommand)} className="h-8 px-2 text-xs">
                      {showAdvancedCommand ? "Hide Advanced" : "Show Advanced"}
                    </Button>
                  </div>
                  <Input id="package-name" placeholder="E.g. mcp-package" value={packageName} onChange={(e) => setPackageName(e.target.value)} />
                  <p className="text-xs text-muted-foreground">The name of the package to execute</p>
                </div>

                {showAdvancedCommand && (
                  <div className="space-y-2 border-l-2 pl-4 ml-2 border-blue-100">
                    <Label htmlFor="command-prefix">Command Prefix (Optional)</Label>
                    <Input id="command-prefix" placeholder="E.g., --from git+https://github.com/user/repo" value={commandPrefix} onChange={(e) => setCommandPrefix(e.target.value)} />
                    <p className="text-xs text-muted-foreground">Additional parameters that go between the executor and package name</p>
                  </div>
                )}

                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Arguments</Label>
                  </div>

                  <div className="space-y-2">
                    {argPairs.map((pair, index) => (
                      <div key={index} className="flex gap-2 items-center">
                        <Input placeholder="Argument (e.g., --directory /path or --verbose, ...)" value={pair.value} onChange={(e) => updateArgPair(index, e.target.value)} className="flex-1" />
                        <Button variant="ghost" size="sm" onClick={() => removeArgPair(index)} disabled={argPairs.length === 1} className="p-1">
                          <Trash2 className="h-4 w-4 text-red-500" />
                        </Button>
                      </div>
                    ))}
                    <Button variant="outline" size="sm" onClick={addArgPair} className="mt-2 w-full">
                      <PlusCircle className="h-4 w-4 mr-2" />
                      Add Argument
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <Label>Environment Variables</Label>
                  </div>

                  <div className="space-y-2">
                    {envPairs.map((pair, index) => (
                      <div key={index} className="flex gap-2 items-center">
                        <Input placeholder="Key (e.g., NODE_ENV)" value={pair.key} onChange={(e) => updateEnvPair(index, "key", e.target.value)} className="flex-1" />
                        <Input placeholder="Value (e.g., production)" value={pair.value} onChange={(e) => updateEnvPair(index, "value", e.target.value)} className="flex-1" />
                        <Button variant="ghost" size="sm" onClick={() => removeEnvPair(index)} disabled={envPairs.length === 1} className="p-1">
                          <Trash2 className="h-4 w-4 text-red-500" />
                        </Button>
                      </div>
                    ))}
                    <Button variant="outline" size="sm" onClick={addEnvPair} className="mt-2 w-full">
                      <PlusCircle className="h-4 w-4 mr-2" />
                      Add Environment Variable
                    </Button>
                  </div>
                </div>

                <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced} className="border rounded-md p-2">
                  <CollapsibleTrigger className="flex w-full items-center justify-between p-2">
                    <span className="font-medium">Advanced Settings</span>
                    {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </CollapsibleTrigger>
                  <CollapsibleContent className="space-y-4 pt-2">
                    <div className="space-y-2">
                      <Label htmlFor="stderr">Stderr Handling</Label>
                      <Input id="stderr" placeholder="e.g., pipe" value={stderr} onChange={(e) => setStderr(e.target.value)} />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="cwd">Working Directory</Label>
                      <Input id="cwd" placeholder="e.g., /path/to/working/dir" value={cwd} onChange={(e) => setCwd(e.target.value)} />
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </TabsContent>

              <TabsContent value="url" className="pt-4 space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="url">Server URL</Label>
                  <Input id="url" placeholder="e.g., https://example.com/mcp-endpoint" value={url} onChange={(e) => setUrl(e.target.value)} />
                  <p className="text-xs text-muted-foreground">Enter the URL of the MCP server endpoint</p>
                </div>

                <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced} className="border rounded-md p-2">
                  <CollapsibleTrigger className="flex w-full items-center justify-between p-2">
                    <span className="font-medium">Advanced Settings</span>
                    {showAdvanced ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                  </CollapsibleTrigger>
                  <CollapsibleContent className="space-y-4 pt-2">
                    <div className="space-y-2">
                      <Label htmlFor="headers">Headers (JSON)</Label>
                      <Input id="headers" placeholder='e.g., {"Authorization": "Bearer token"}' value={headers} onChange={(e) => setHeaders(e.target.value)} />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="timeout">Connection Timeout (seconds)</Label>
                      <Input id="timeout" type="number" value={timeout} onChange={(e) => setTimeout(e.target.value)} />
                      <p className="text-xs text-muted-foreground">Default: 5 seconds</p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="sse-read-timeout">SSE Read Timeout (seconds)</Label>
                      <Input id="sse-read-timeout" type="number" value={sseReadTimeout} onChange={(e) => setSseReadTimeout(e.target.value)} />
                      <p className="text-xs text-muted-foreground">Default: 300 seconds (5 minutes)</p>
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              </TabsContent>
            </Tabs>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleClose} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={isSubmitting} className="bg-blue-500 hover:bg-blue-600 text-white">
            {isSubmitting ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Adding...
              </>
            ) : (
              "Add Server"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}