from format_demo_conch import ExampleNode


node = ExampleNode()
node.init()
node.read()
results = node.execute()

print(results[0])