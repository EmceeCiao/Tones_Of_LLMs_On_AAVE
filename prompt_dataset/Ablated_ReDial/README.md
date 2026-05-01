# About 

This Ablated_ReDial folder will contain a collection of prompts from the original ReDial dataset located in Orig_ReDial/redial_gold. 

Orig_ReDial/redial_gold contains json lists split between 4 categories and a json of the ids. 

The layout of the algorithm JSON file is the following: 
{   vanilla: 
        original:
            { 
                prompt: 
                data_name: 
                function_name: 
                task_idx:  
            } 
        AAVE:  
            {
                prompt: 
                data_name: 
                function_name: 
                task_idx:  
            }
    cot:  
        original:  
            {  
               See above: 
            }
        AAVE:    
            { 
                See above: 
            } 
} 

We are only considering the algorithm JSON file as it's the closest to an user prompt rather than looking at comprehensive, logic, and math. 

Interesting enough, something important to keep in mind is that this dataset can still be used as a benchmark reference (unless contaimination has occured which may be possible given the code being public on GitHub, but maybe that can be caught?)   

Anyways the pattern we will be using in our Ablated_ReDial will be  
{ 
    vanilla_prompt: 
    aave_prompt: 
    function_name: 
    task_idx:  
} 

As of this moment, 8 function have been ablated and these were chosen keeping in mind that not every participant will have studied computer science. 





