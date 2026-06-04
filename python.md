# Python Cheat Sheet

## Data Types
- int, float, str, bool, list, tuple, dict, set

## Variables
x = 10
name = "Harsha"
is_active = True

## Strings
name.upper()          # HARSHA
name.lower()          # harsha
name.strip()          # remove whitespace
f"Hello {name}"       # f-string formatting

## Lists
fruits = ["apple", "banana", "cherry"]
fruits.append("mango")   # add to end
fruits.pop()              # remove last item
fruits[0]                 # access first item
len(fruits)               # length

## Dictionaries
person = {"name": "Harsha", "age": 25}
person["name"]            # access value
person["city"] = "Brisbane"  # add key
person.keys()             # all keys
person.values()           # all values

## Loops
for fruit in fruits:
    print(fruit)

for i in range(5):
    print(i)

## Functions
def greet(name):
    return f"Hello, {name}!"

## Conditionals
if x > 5:
    print("big")
elif x == 5:
    print("five")
else:
    print("small")

## List Comprehensions
squares = [x**2 for x in range(10)]

## Common Built-ins
print(), len(), range(), type(), int(), str(), float(), input()