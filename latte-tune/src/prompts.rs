use latte_ai::models::{Message, Role};

/// A test prompt to evaluate model generation quality.
pub struct TestPrompt {
    pub name: &'static str,
    pub system: &'static str,
    pub user: &'static str,
    /// Category for organizing results.
    pub category: PromptCategory,
    /// Expected characteristics we want to observe.
    pub description: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PromptCategory {
    CodeGeneration,
    Debugging,
    Refactoring,
    Explanation,
    Architecture,
}

impl std::fmt::Display for PromptCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptCategory::CodeGeneration => write!(f, "code-gen"),
            PromptCategory::Debugging => write!(f, "debug"),
            PromptCategory::Refactoring => write!(f, "refactor"),
            PromptCategory::Explanation => write!(f, "explain"),
            PromptCategory::Architecture => write!(f, "architecture"),
        }
    }
}

/// Returns a set of programming-focused test prompts.
pub fn all_prompts() -> Vec<TestPrompt> {
    vec![
        // ── Code Generation ────────────────────────────────────────────
        TestPrompt {
            name: "rust-parse-json",
            category: PromptCategory::CodeGeneration,
            system: "You are an expert Rust programmer.",
            user: r#"Write a Rust function that:
1. Reads a JSON file at a given path
2. Parses it into a Vec<Record> where each Record has: id (u64), name (String), tags (Vec<String>)
3. Filters records where tags include "active"
4. Returns the filtered records sorted by id ascending

Include proper error handling with anyhow."#,
            description: "Checks code structure, error handling, and idiomatic Rust",
        },
        TestPrompt {
            name: "typescript-react-component",
            category: PromptCategory::CodeGeneration,
            system: "You are a senior TypeScript/React developer.",
            user: r#"Create a React component `DataTable<T>` that:
- Accepts columns definition and data array as props
- Supports sorting by clicking column headers
- Supports filtering by a text input
- Renders with proper TypeScript generics
- Uses CSS modules or inline styles (no external library)

Include the TypeScript types for Column<T>."#,
            description: "Checks TypeScript generics, React patterns, and completeness",
        },
        TestPrompt {
            name: "python-data-pipeline",
            category: PromptCategory::CodeGeneration,
            system: "You are an expert Python developer focused on data engineering.",
            user: r#"Write a Python class `DataPipeline` that:
1. Takes a list of CSV file paths on initialization
2. Has a method `extract()` that reads all CSVs into DataFrames
3. Has a method `transform()` that:
   - Removes duplicate rows
   - Fills NaN values with column means for numeric columns
   - Adds a `processed_at` timestamp column
4. Has a method `load(db_url: str)` that writes to a SQLite database
5. Has proper type hints and docstrings
6. Includes basic logging

Use pandas and sqlalchemy."#,
            description: "Checks code organization, library usage, and completeness",
        },

        // ── Debugging ──────────────────────────────────────────────────
        TestPrompt {
            name: "debug-memory-leak",
            category: PromptCategory::Debugging,
            system: "You are a systems programming expert.",
            user: r#"I have a Rust program that processes large files and seems to have a memory leak.
The code below reads a file line by line, but memory usage keeps growing.

```rust
use std::fs::File;
use std::io::{BufRead, BufReader};

fn process_file(path: &str) -> Vec<String> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut results = Vec::new();
    
    for line in reader.lines() {
        let line = line.unwrap();
        let processed = line.trim().to_string();
        results.push(processed);
    }
    
    results
}
```

What's wrong and how would you fix it?"#,
            description: "Checks ability to identify memory issues and propose fixes",
        },
        TestPrompt {
            name: "debug-null-pointer",
            category: PromptCategory::Debugging,
            system: "You are a senior software engineer.",
            user: r#"This TypeScript function sometimes throws "Cannot read properties of null".
Can you find the bug and fix it?

```typescript
interface Config {
  database?: {
    host?: string;
    port?: number;
    credentials?: {
      username?: string;
      password?: string;
    };
  };
  cache?: {
    ttl?: number;
    provider?: string;
  };
}

function getConfigValue(config: Config, path: string): string {
  const parts = path.split('.');
  let current: any = config;
  
  for (const part of parts) {
    current = current[part];
  }
  
  return current;
}
```"#,
            description: "Checks defensive programming and null-safety awareness",
        },

        // ── Refactoring ────────────────────────────────────────────────
        TestPrompt {
            name: "refactor-spaghetti",
            category: PromptCategory::Refactoring,
            system: "You are a code quality expert.",
            user: r#"Refactor this Go function to be more readable, testable, and maintainable:

```go
func ProcessData(data []byte, t string) ([]byte, error) {
    var result []byte
    if t == "json" {
        var m map[string]interface{}
        json.Unmarshal(data, &m)
        if v, ok := m["name"]; ok {
            result = []byte(fmt.Sprintf("Name: %v", v))
        } else {
            result = []byte("Name: unknown")
        }
    } else if t == "csv" {
        r := csv.NewReader(strings.NewReader(string(data)))
        records, _ := r.ReadAll()
        if len(records) > 0 {
            result = []byte(fmt.Sprintf("Rows: %d", len(records)))
        } else {
            result = []byte("No data")
        }
    } else if t == "xml" {
        // similar XML handling...
        result = []byte("XML processed")
    } else {
        return nil, fmt.Errorf("unknown type: %s", t)
    }
    return result, nil
}
```"#,
            description: "Checks refactoring quality, pattern recognition, and code organization",
        },

        // ── Explanation ────────────────────────────────────────────────
        TestPrompt {
            name: "explain-concurrency",
            category: PromptCategory::Explanation,
            system: "You are a computer science educator.",
            user: r#"Explain the difference between these three concurrency patterns in Rust:
1. `tokio::spawn`
2. `std::thread::spawn` 
3. `rayon::par_iter`

When would you use each one? Give a concrete example for each."#,
            description: "Checks explanatory quality, accuracy, and example relevance",
        },

        // ── Architecture ───────────────────────────────────────────────
        TestPrompt {
            name: "design-microservice",
            category: PromptCategory::Architecture,
            system: "You are a software architect.",
            user: r#"Design a simple URL shortening service.

Requirements:
- Create short URLs from long URLs
- Redirect from short URL to long URL
- Track visit counts per URL
- REST API

Please provide:
1. API endpoints design
2. Database schema
3. Key implementation details in Rust (using axum)
4. How you'd handle scaling"#,
            description: "Checks architectural thinking, API design, and systems knowledge",
        },
    ]
}

impl TestPrompt {
    /// Build messages for this test prompt.
    pub fn to_messages(&self) -> Vec<Message> {
        vec![
            Message::text(Role::System, self.system.to_string()),
            Message::text(Role::User, self.user.to_string()),
        ]
    }
}
