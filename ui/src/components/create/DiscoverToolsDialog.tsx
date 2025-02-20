import { useState } from "react";
import { getBackendUrl } from "@/lib/utils";
import { Tool } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Loader2, Terminal, Globe } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";

interface DiscoverToolsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onShowSelectTools: (tools: Tool[]) => void;
}

interface StdioServerParameters {
  /**
   * The executable to run to start the server.
   */
  command: string;
  /**
   * Command line arguments to pass to the executable.
   */
  args?: string[];
  /**
   * The environment to use when spawning the process.
   */
  env?: Record<string, string>;
  /**
   * How to handle stderr of the child process.
   */
  stderr?: string | number;
  /**
   * The working directory to use when spawning the process.
   */
  cwd?: string;
}

interface SseServerParams {
  url: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  headers?: Record<string, any>;
  timeout?: number;
  sse_read_timeout?: number;
}

export const DiscoverToolsDialog = ({ open, onOpenChange, onShowSelectTools }: DiscoverToolsDialogProps) => {
  const [activeTab, setActiveTab] = useState<"command" | "url">("command");

  // Command tab state
  const [command, setCommand] = useState("");
  const [args, setArgs] = useState("");
  const [envVars, setEnvVars] = useState("");
  const [stderrType, setStderrType] = useState("inherit");
  const [cwd, setCwd] = useState("");

  // URL tab state
  const [url, setUrl] = useState("");
  const [headers, setHeaders] = useState("");
  const [timeout, setTimeout] = useState("5");
  const [sseReadTimeout, setSseReadTimeout] = useState("300");

  // Shared state
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [error, setError] = useState("");

  const resetState = () => {
    setCommand("");
    setArgs("");
    setEnvVars("");
    setStderrType("inherit");
    setCwd("");
    setUrl("");
    setHeaders("");
    setTimeout("5");
    setSseReadTimeout("300");
    setError("");
    setIsDiscovering(false);
  };

  const handleClose = () => {
    resetState();
    onOpenChange(false);
  };

  const parseEnvVars = (input: string): Record<string, string> => {
    const result: Record<string, string> = {};
    if (!input.trim()) return result;

    try {
      return JSON.parse(input);
    } catch (e) {
      console.error(e);
      // If not valid JSON, try to parse KEY=VALUE format
      const lines = input.split("\n").filter((line) => line.trim());
      for (const line of lines) {
        const [key, ...valueParts] = line.split("=");
        if (key && valueParts.length) {
          result[key.trim()] = valueParts.join("=").trim();
        }
      }
      return result;
    }
  };

  const parseHeaders = (input: string): Record<string, string> => {
    return parseEnvVars(input); // Same parsing logic can be used
  };

  const parseArguments = (input: string): string[] => {
    if (!input.trim()) return [];

    try {
      const parsed = JSON.parse(input);
      return Array.isArray(parsed) ? parsed : [input];
    } catch {
      // If not valid JSON, split by spaces or newlines
      return input.split(/[\s\n]+/).filter((arg) => arg.trim());
    }
  };

  const discoverTools = async () => {
    try {
      setIsDiscovering(true);
      setError("");

      let payload;
      if (activeTab === "command") {
        if (!command.trim()) {
          setError("Command is required");
          return;
        }

        const serverParams: StdioServerParameters = {
          command: command.trim(),
        };

        // Only add non-empty parameters
        const parsedArgs = parseArguments(args);
        if (parsedArgs.length > 0) {
          serverParams.args = parsedArgs;
        }

        const parsedEnv = parseEnvVars(envVars);
        if (Object.keys(parsedEnv).length > 0) {
          serverParams.env = parsedEnv;
        }

        if (stderrType && stderrType !== "inherit") {
          serverParams.stderr = stderrType;
        }

        if (cwd.trim()) {
          serverParams.cwd = cwd.trim();
        }

        payload = {
          type: "stdio",
          server_params: serverParams,
        };
      } else {
        if (!url.trim()) {
          setError("URL is required");
          return;
        }

        const serverParams: SseServerParams = {
          url: url.trim(),
          timeout: parseFloat(timeout) || 5,
          sse_read_timeout: parseFloat(sseReadTimeout) || 300,
        };

        const parsedHeaders = parseHeaders(headers);
        if (Object.keys(parsedHeaders).length > 0) {
          serverParams.headers = parsedHeaders;
        }

        payload = {
          type: "sse",
          server_params: serverParams,
        };
      }

      const response = await fetch(`${getBackendUrl()}/tools/discover`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Failed to discover tools");
      }

      const discoveredToolsData: Tool[] = await response.json();

      if (discoveredToolsData.length > 0) {
        // Close this dialog and open the select tools dialog
        handleClose();
        onShowSelectTools(discoveredToolsData);
      } else {
        setError("No tools were discovered. Please check your input and try again.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unknown error occurred");
    } finally {
      setIsDiscovering(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-[#2A2A2A] border-[#3A3A3A] text-white max-w-4xl">
        <DialogHeader>
          <DialogTitle>Discover MCP Tools</DialogTitle>
        </DialogHeader>

        <div className="py-4">
          <Tabs defaultValue="command" value={activeTab} onValueChange={(value) => setActiveTab(value as "command" | "url")} className="w-full">
            <TabsList className="w-full bg-[#1A1A1A]">
              <TabsTrigger value="command" className="flex-1 data-[state=active]:bg-violet-500">
                <Terminal className="h-4 w-4 mr-2" />
                Command
              </TabsTrigger>
              <TabsTrigger value="url" className="flex-1 data-[state=active]:bg-violet-500">
                <Globe className="h-4 w-4 mr-2" />
                URL
              </TabsTrigger>
            </TabsList>

            <TabsContent value="command" className="mt-4 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="command">
                  Command <span className="text-red-500">*</span>
                </Label>
                <Input id="command" placeholder="e.g. npx, uvx, uv" value={command} onChange={(e) => setCommand(e.target.value)} className="bg-[#1A1A1A] border-[#3A3A3A]" />
              </div>

              <div className="space-y-2">
                <Label htmlFor="args">Arguments (one per line or as JSON array)</Label>
                <Input id="args" placeholder="e.g. mcp-server-kubernetes" value={args} onChange={(e) => setArgs(e.target.value)} className="bg-[#1A1A1A] border-[#3A3A3A]" />
              </div>

              <div className="space-y-2">
                <Label htmlFor="env">Environment Variables (KEY=VALUE format or JSON)</Label>
                <textarea
                  id="env"
                  placeholder="API_KEY=your_key_here&#10;DEBUG=true"
                  value={envVars}
                  onChange={(e) => setEnvVars(e.target.value)}
                  className="w-full min-h-[100px] rounded-md bg-[#1A1A1A] border border-[#3A3A3A] px-3 py-2 text-sm"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="stderr">Stderr Handling</Label>
                <select id="stderr" value={stderrType} onChange={(e) => setStderrType(e.target.value)} className="w-full bg-[#1A1A1A] border-[#3A3A3A] rounded-md px-3 py-2 text-sm">
                  <option value="inherit">inherit</option>
                  <option value="pipe">pipe</option>
                  <option value="ignore">ignore</option>
                </select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="cwd">Working Directory (optional)</Label>
                <Input id="cwd" placeholder="e.g. /path/to/working/directory" value={cwd} onChange={(e) => setCwd(e.target.value)} className="bg-[#1A1A1A] border-[#3A3A3A]" />
              </div>
            </TabsContent>

            <TabsContent value="url" className="mt-4 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="url">
                  URL <span className="text-red-500">*</span>
                </Label>
                <Input id="url" placeholder="https://example.com/mcp-tools" value={url} onChange={(e) => setUrl(e.target.value)} className="bg-[#1A1A1A] border-[#3A3A3A]" />
              </div>

              <div className="space-y-2">
                <Label htmlFor="headers">Headers (KEY=VALUE format or JSON)</Label>
                <textarea
                  id="headers"
                  placeholder="Authorization=Bearer token&#10;Content-Type=application/json"
                  value={headers}
                  onChange={(e) => setHeaders(e.target.value)}
                  className="w-full min-h-[100px] rounded-md bg-[#1A1A1A] border border-[#3A3A3A] px-3 py-2 text-sm"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="timeout">Timeout (seconds)</Label>
                  <Input id="timeout" type="number" min="1" step="1" placeholder="5" value={timeout} onChange={(e) => setTimeout(e.target.value)} className="bg-[#1A1A1A] border-[#3A3A3A]" />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="sse_read_timeout">SSE Read Timeout (seconds)</Label>
                  <Input
                    id="sse_read_timeout"
                    type="number"
                    min="1"
                    step="1"
                    placeholder="300"
                    value={sseReadTimeout}
                    onChange={(e) => setSseReadTimeout(e.target.value)}
                    className="bg-[#1A1A1A] border-[#3A3A3A]"
                  />
                </div>
              </div>
            </TabsContent>
          </Tabs>

          {error && (
            <Alert variant="destructive" className="mt-4">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>

        <DialogFooter>
          <div className="flex justify-end gap-2 w-full">
            <Button variant="ghost" onClick={handleClose} disabled={isDiscovering}>
              Cancel
            </Button>
            <Button className="bg-violet-500 hover:bg-violet-600" onClick={discoverTools} disabled={isDiscovering}>
              {isDiscovering ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Discovering...
                </>
              ) : (
                "Discover Tools"
              )}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
