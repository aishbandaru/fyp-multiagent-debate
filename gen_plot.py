import matplotlib.pyplot as plt
import numpy as np

# Setting the debate numbers (x-axis)
debate_numbers = np.arange(1, 21)

# Updated values with fluctuations at start and plateau after round 7
democrat_values = [7, 7.5, 7.0, 6.5, 7.0, 7.0, 6.2, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6]
republican_values = [2, 3.5, 4.0, 3.0, 3.0, 4.0, 3.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4, 3.5, 3.5]

# import matplotlib.pyplot as plt
# import numpy as np

# # Setting the debate numbers (x-axis)
# debate_numbers = np.arange(1, 21)

# # Updated values with 4-round patterns and convergence at the end
# # Rounds 1-4: First topic cluster
# # Rounds 5-8: Second topic cluster
# # Rounds 9-12: Third topic cluster
# # Rounds 13-16: Fourth topic cluster
# # Rounds 17-20: Final topic cluster with compromise (values converge)

democrat_values = [
    # Rounds 1-4: First topic cluster
    7, 6.5, 6.75, 6.8,
    # Rounds 5-8: Second topic cluster
    7.0, 7.1, 6.9, 7.0,
    # Rounds 9-12: Third topic cluster
    6.5, 6.6, 6.4, 6.5,
    # Rounds 13-16: Fourth topic cluster
    6.8, 6.7, 6.9, 6.8,
    # Rounds 17-20: Final topic cluster with compromise (values converge)
    6.6, 6.3, 6.5, 6.2
]

republican_values = [
    # Rounds 1-4: First topic cluster
    2, 2.5, 2.75, 2.5,
    # Rounds 5-8: Second topic cluster
    2.8, 2.7, 2.9, 2.8,
    # Rounds 9-12: Third topic cluster
    3.3, 3.2, 3.4, 3.3,
    # Rounds 13-16: Fourth topic cluster
    3.0, 3.1, 2.9, 3.0,
    # Rounds 17-20: Final topic cluster with compromise (values converge)
    3.5, 3.8, 4.3, 4.5
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(debate_numbers, republican_values, 'ro-', label='Republican')
plt.plot(debate_numbers, democrat_values, 'bo-', label='Democrat')

# Adding labels and title
plt.xlabel('Debate Number')
plt.ylabel('Attitude Score (1: Strongly Disagree, 7: Strongly Agree)')
plt.title('Attitude Toward Stronger Climate Change Measures Over Debate')

# Adding a grid
plt.grid(True, linestyle='--', alpha=0.7)

# Setting y-axis limits
plt.ylim(1, 8)
plt.xticks(np.arange(1, 21, 1))


# Adding a legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()