<script>
	var userChoice = prompt("Do you choose rock, paper or scissors?");
	var computerChoice = Math.random();
	if (computerChoice < 0.34) {
		computerChoice = "rock";
	} else if(computerChoice <= 0.67) {
		computerChoice = "paper";
	} else {
		computerChoice = "scissors";
	}

	var compare = function(choice1, choice2) {
	    console.log("You chose: " + userChoice);

	    if (choice1 === "rock" || choice1 === "scissors" || choice1 === "paper")
	    {
	        console.log("Computer: " + computerChoice);
	        if (choice1 === choice2)
	            return "The result is a tie!";
	        else
	            if (choice1 === "rock")
	                if (choice2 === "scissors")
	                    return "rock wins!";
	                else
	                    return "paper wins";
	            else
	                if (choice1 === "paper")
	                    if (choice2 === "rock")
	                        return "paper wins!";
	                    else
	                        return "scissors wins!";
	                else
	                    if (choice1 === "scissors")
	                        if (choice2 === "rock")
	                            return "rock wins!";
	                        else
	                            return "paper wins!";
	    }
	    else
	    {
	        userChoice = prompt("Do you choose rock, paper or scissors?");
	        return compare(userChoice, computerChoice);
	    }

	};

	console.log(compare(userChoice, computerChoice));

</script>
