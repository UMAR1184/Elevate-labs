def add(a, b):
    return a + b
def subtract(a, b):
    return a - b
def multiply(a, b):
    return a * b
def divide(a, b):
    if b == 0:
        return "Error: Division by zero is undefined."
    return a / b
def get_input():
    try:
        a = float(input("Enter first number: "))
        b = float(input("Enter second number: "))
        return a, b
    except ValueError:
        print("Invalid input! Please enter numeric values.")
        return get_input()
def main():
    while True:
        print("\nSimple CLI Calculator")
        print("1. Add")
        print("2. Subtract")
        print("3. Multiply")
        print("4. Divide")
        print("5. Exit")
        choice = input("Choose operation (1-5): ")
        if choice == '5':
            print("Exiting calculator. Goodbye!")
            break
        if choice in ('1', '2', '3', '4'):
            a, b = get_input()
            if choice == '1':
                print(f"Result: {add(a, b)}")
            elif choice == '2':
                print(f"Result: {subtract(a, b)}")
            elif choice == '3':
                print(f"Result: {multiply(a, b)}")
            elif choice == '4':
                result = divide(a, b)
                print(f"Result: {result}")
        else:
            print("Invalid choice. Please choose a valid operation.")
if __name__ == "__main__":
    main()
